"""Data ingestion, schema normalization, global near-duplicate clustering, and group-aware splitting for PyCode-Vul."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

DATASET_REPO = "S-AIR-L/PyCode-Vul"
TRAIN_FILE = "PyCode_Vul- train-set.csv"
TEST_FILE = "PyCode_Vul test-set.csv"
RANDOM_SEED = 42

# Repositories held out entirely for cross-project out-of-distribution evaluation
HELD_OUT_REPOS = {
    "pallets/flask",
    "PyCQA/bandit",
    "paramiko/paramiko",
    "psf/requests",
    "yaml/pyyaml",
}


class UnionFind:
    """Disjoint-set data structure for grouping near-duplicate code clusters."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, i: int) -> int:
        if self.parent.setdefault(i, i) != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def download_raw_files(cache_dir: Optional[str] = None) -> Tuple[Path, Path]:
    """Download raw dataset CSVs from Hugging Face Hub."""
    train_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=TRAIN_FILE,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    test_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=TEST_FILE,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return Path(train_path), Path(test_path)


def load_raw_combined() -> pd.DataFrame:
    """Load both upstream files and normalize columns preserving provenance metadata."""
    train_path, test_path = download_raw_files()
    df_raw_train = pd.read_csv(train_path)
    df_raw_test = pd.read_csv(test_path)

    # For label == 1: vulnerable_function_source
    # For label == 0: patched_function_source (fallback to vulnerable_function_source if null)
    train_code = df_raw_train.apply(
        lambda r: (
            r["vulnerable_function_source"]
            if r["label"] == 1 or pd.isna(r.get("patched_function_source"))
            else r["patched_function_source"]
        ),
        axis=1,
    )

    df_train = pd.DataFrame(
        {
            "code": train_code,
            "label": df_raw_train["label"].astype(int),
            "repo": df_raw_train.get("repo", "unknown").fillna("unknown"),
            "sha": df_raw_train.get("sha", "").fillna(""),
            "file_path": df_raw_train.get("file_path", "").fillna(""),
            "cwe_ids": df_raw_train.get("cwe_ids", "").fillna(""),
        }
    )

    df_test = pd.DataFrame(
        {
            "code": df_raw_test["function_code"],
            "label": df_raw_test["class"].astype(int),
            "repo": df_raw_test.get("repo", "unknown").fillna("unknown"),
            "sha": df_raw_test.get("sha", "").fillna(""),
            "file_path": df_raw_test.get("file_path", "").fillna(""),
            "cwe_ids": df_raw_test.get("cwe_ids", "").fillna(""),
        }
    )

    combined = pd.concat([df_train, df_test], ignore_index=True)
    combined = combined[combined["code"].notna() & combined["code"].apply(lambda x: isinstance(x, str))].copy()
    combined["code"] = combined["code"].str.strip()
    combined = combined[combined["code"].str.len() > 10].reset_index(drop=True)

    # Deduplicate exact identical code strings, prioritizing label=1 if conflict exists
    combined = combined.sort_values(by=["label"], ascending=False)
    combined = combined.drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)

    return combined


def build_global_clusters(df: pd.DataFrame, similarity_threshold: float = 0.80) -> pd.DataFrame:
    """Cluster functions by commit SHA AND global near-duplicate token similarity (>=0.80).

    Guarantees vulnerable functions and their patched twins belong to the exact same cluster.
    """
    df = df.copy()
    n = len(df)
    uf = UnionFind()

    # Step A: Union by commit SHA within repo
    sha_map: Dict[str, List[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        key = f"{row['repo']}::{row['sha']}"
        sha_map[key].append(idx)

    for indices in sha_map.values():
        if len(indices) > 1:
            first = indices[0]
            for other in indices[1:]:
                uf.union(first, other)

    # Step B: Global TF-IDF cosine / token near-duplicate clustering across the whole corpus
    print("Computing global token sparse similarity matrix...")
    vec = TfidfVectorizer(token_pattern=r"[a-zA-Z_]\w*", min_df=2, norm="l2")
    X = vec.fit_transform(df["code"])

    batch_size = 2000
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sim = X[start:end] @ X.T
        rows, cols = (sim >= similarity_threshold).nonzero()
        for r, c in zip(rows, cols):
            i = start + r
            if i != c:
                uf.union(i, c)

    df["cluster_id"] = [uf.find(i) for i in range(n)]
    return df


def verify_zero_crossing_pairs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    similarity_threshold: float = 0.80,
) -> int:
    """Count how many near-duplicate pairs (similarity >= 0.80) cross split boundaries."""
    corpus = list(train_df["code"]) + list(test_df["code"])
    n_train = len(train_df)
    n_test = len(test_df)
    n_val = 0
    if val_df is not None:
        corpus += list(val_df["code"])
        n_val = len(val_df)

    vec = TfidfVectorizer(token_pattern=r"[a-zA-Z_]\w*", min_df=2, norm="l2")
    X = vec.fit_transform(corpus)

    X_train = X[:n_train]
    X_test = X[n_train : n_train + n_test]

    # Test vs Train crossings
    sim_test_train = X_test @ X_train.T
    test_crossings = int((sim_test_train >= similarity_threshold).nnz)

    val_crossings = 0
    if val_df is not None:
        X_val = X[n_train + n_test :]
        sim_val_train = X_val @ X_train.T
        val_crossings = int((sim_val_train >= similarity_threshold).nnz)

    return test_crossings + val_crossings


def make_group_aware_split(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-aware random split by cluster_id (70% train, 15% val, 15% test)."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["cluster_id"]))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["cluster_id"]))

    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def make_repo_holdout_split(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repository-level holdout split: entire unseen repos held out in test set."""
    # Find all cluster IDs that touch any held out repo
    held_out_clusters = set(df[df["repo"].isin(HELD_OUT_REPOS)]["cluster_id"])

    test_mask = df["cluster_id"].isin(held_out_clusters)
    test_df = df[test_mask].copy().reset_index(drop=True)
    rem_df = df[~test_mask].copy().reset_index(drop=True)

    # Split remaining clusters into 80% train / 20% val by cluster_id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, val_idx = next(gss.split(rem_df, groups=rem_df["cluster_id"]))

    train_df = rem_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = rem_df.iloc[val_idx].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def prepare_and_save_data(output_dir: str = "ml/data", seed: int = RANDOM_SEED) -> None:
    """Prepares and saves both the group-wise split and the repo-holdout split."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading raw dataset and standardizing...")
    combined = load_raw_combined()
    print(f"Total deduplicated samples: {len(combined)}")

    print("Step 2: Building commit and global near-duplicate twin clusters (threshold >= 0.80)...")
    clustered = build_global_clusters(combined, similarity_threshold=0.80)
    n_clusters = clustered["cluster_id"].nunique()
    print(f"Total unique near-duplicate/commit clusters: {n_clusters}")

    print("\n--- Generating Split A: Random Group-Wise Split ---")
    tr_gw, val_gw, te_gw = make_group_aware_split(clustered, seed=seed)
    crossings_gw = verify_zero_crossing_pairs(tr_gw, te_gw, val_gw, similarity_threshold=0.80)
    print(f"Group-Wise Train size: {len(tr_gw)}, Val size: {len(val_gw)}, Test size: {len(te_gw)}")
    print(f"Near-duplicate pairs crossing boundaries: {crossings_gw}")
    assert crossings_gw == 0, f"Expected 0 crossing pairs, got {crossings_gw}"

    tr_gw.to_csv(out_path / "train_group.csv", index=False)
    val_gw.to_csv(out_path / "val_group.csv", index=False)
    te_gw.to_csv(out_path / "test_group.csv", index=False)

    print("\n--- Generating Split B: Held-Out Repositories Split ---")
    tr_rh, val_rh, te_rh = make_repo_holdout_split(clustered, seed=seed)
    crossings_rh = verify_zero_crossing_pairs(tr_rh, te_rh, val_rh, similarity_threshold=0.80)
    print(f"Repo-Holdout Train size: {len(tr_rh)}, Val size: {len(val_rh)}, Test size: {len(te_rh)}")
    print(f"Held-out test repos: {sorted(list(HELD_OUT_REPOS))}")
    print(f"Near-duplicate pairs crossing boundaries: {crossings_rh}")
    assert crossings_rh == 0, f"Expected 0 crossing pairs in repo holdout, got {crossings_rh}"

    tr_rh.to_csv(out_path / "train_repo_holdout.csv", index=False)
    val_rh.to_csv(out_path / "val_repo_holdout.csv", index=False)
    te_rh.to_csv(out_path / "test_repo_holdout.csv", index=False)

    # Set default train.csv, val.csv, test.csv to the group-wise split
    tr_gw.to_csv(out_path / "train.csv", index=False)
    val_gw.to_csv(out_path / "val.csv", index=False)
    te_gw.to_csv(out_path / "test.csv", index=False)
    print(f"\nDefault splits successfully written to {out_path}/ with ZERO crossing pairs!")


if __name__ == "__main__":
    prepare_and_save_data()
