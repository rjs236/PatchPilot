"""Unit tests for AST-based code analyzer rules."""

import pytest
from patchpilot.analyzers.ast_analyzer import analyze_ast


def test_clean_code_produces_zero_ast_findings():
    code = """
def add(a: int, b: int) -> int:
    \"\"\"Add two integers.\"\"\"
    return a + b
"""
    findings = analyze_ast(code)
    assert len(findings) == 0


def test_mutable_default_args():
    code = """
def process_data(items=[], config={}):
    items.append(config)
    return items
"""
    findings = analyze_ast(code)
    mutable_findings = [f for f in findings if f.rule_id == "mutable-default-arg"]
    assert len(mutable_findings) >= 2
    assert mutable_findings[0].category == "bug"
    assert mutable_findings[0].severity == "medium"
    assert "mutable default argument" in mutable_findings[0].message


def test_bare_except():
    code = """
try:
    x = 1 / 0
except:
    pass
"""
    findings = analyze_ast(code)
    bare = [f for f in findings if f.rule_id == "bare-except"]
    assert len(bare) == 1
    assert bare[0].category == "bug"
    assert bare[0].severity == "medium"


def test_bare_except_base_exception():
    code = """
try:
    x = 1
except BaseException:
    pass
"""
    findings = analyze_ast(code)
    bare = [f for f in findings if f.rule_id == "bare-except"]
    assert len(bare) == 1


def test_eval_exec():
    code = """
def run_dynamic(cmd):
    eval(cmd)
    exec(cmd)
"""
    findings = analyze_ast(code)
    sec_findings = [f for f in findings if f.rule_id == "eval-exec"]
    assert len(sec_findings) == 2
    assert all(f.category == "security" for f in sec_findings)
    assert all(f.severity == "high" for f in sec_findings)


def test_comparison_to_none():
    code = """
def check(val):
    if val == None:
        return False
    if None != val:
        return True
    return None
"""
    findings = analyze_ast(code)
    none_findings = [f for f in findings if f.rule_id == "comparison-to-none"]
    assert len(none_findings) == 2
    assert all(f.category == "style" for f in none_findings)
    assert all(f.severity == "low" for f in none_findings)


def test_cyclomatic_complexity_medium_and_high():
    # Construct code with McCabe complexity > 10 and another > 15
    code_medium = """
def medium_comp(x):
    if x == 1: pass
    elif x == 2: pass
    elif x == 3: pass
    elif x == 4: pass
    elif x == 5: pass
    elif x == 6: pass
    elif x == 7: pass
    elif x == 8: pass
    elif x == 9: pass
    elif x == 10: pass
    return x
"""
    findings = analyze_ast(code_medium)
    med = [f for f in findings if f.rule_id == "cyclomatic-complexity-medium"]
    assert len(med) == 1
    assert med[0].severity == "medium"

    code_high = """
def high_comp(x):
    if x == 1: pass
    elif x == 2: pass
    elif x == 3: pass
    elif x == 4: pass
    elif x == 5: pass
    elif x == 6: pass
    elif x == 7: pass
    elif x == 8: pass
    elif x == 9: pass
    elif x == 10: pass
    elif x == 11: pass
    elif x == 12: pass
    elif x == 13: pass
    elif x == 14: pass
    elif x == 15: pass
    return x
"""
    findings_high = analyze_ast(code_high)
    high = [f for f in findings_high if f.rule_id == "cyclomatic-complexity-high"]
    assert len(high) == 1
    assert high[0].severity == "high"


def test_deep_nesting():
    code = """
def nested():
    if True:
        for i in range(1):
            while True:
                with open("x") as f:
                    if True:
                        print("too deep")
"""
    findings = analyze_ast(code)
    deep = [f for f in findings if f.rule_id == "deep-nesting"]
    assert len(deep) >= 1
    assert deep[0].category == "complexity"
    assert deep[0].severity == "medium"


def test_long_function():
    lines = ["def very_long_function():"]
    for i in range(55):
        lines.append(f"    a_{i} = {i}")
    lines.append("    return a_0")
    code = "\n".join(lines)

    findings = analyze_ast(code)
    long_f = [f for f in findings if f.rule_id == "long-function"]
    assert len(long_f) == 1
    assert long_f[0].category == "complexity"
    assert long_f[0].severity == "medium"


def test_too_many_args():
    code = """
def configure(a, b, c, d, e, f, g):
    return a + b + c + d + e + f + g
"""
    findings = analyze_ast(code)
    arg_f = [f for f in findings if f.rule_id == "too-many-args"]
    assert len(arg_f) == 1
    assert arg_f[0].category == "complexity"
    assert arg_f[0].severity == "low"


def test_syntax_error_returns_high_severity_and_never_crashes():
    code = "def broken_syntax(:\n    pass"
    findings = analyze_ast(code)
    assert len(findings) == 1
    assert findings[0].rule_id == "syntax-error"
    assert findings[0].severity == "high"
    assert findings[0].category == "bug"
    assert findings[0].line >= 1


def test_indented_snippet_is_dedented_and_parsed_successfully():
    # An indented method pasted directly from inside a class
    code = "    def handle_request(self, target, payload=[]):\n        eval(payload)\n"
    findings = analyze_ast(code)
    # Should not produce syntax-error; should successfully detect mutable default and eval
    rule_ids = {f.rule_id for f in findings}
    assert "syntax-error" not in rule_ids
    assert "mutable-default-arg" in rule_ids
    assert "eval-exec" in rule_ids

