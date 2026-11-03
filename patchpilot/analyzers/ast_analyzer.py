"""AST-based code analyzer implementing custom Python checks."""

import ast
import textwrap
from typing import Any, List, Optional

from patchpilot.analyzers.models import Finding


class ASTAnalyzer(ast.NodeVisitor):
    """Inspects Python Abstract Syntax Trees for bugs, security risks, complexity, and style."""

    def __init__(self, filename: str = "<string>") -> None:
        self.filename = filename
        self.findings: List[Finding] = []
        self._nesting_depth = 0
        self._reported_nesting_nodes: set[int] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function_definition(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function_definition(node)
        self.generic_visit(node)

    def _check_function_definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # 1. Long functions (>50 lines)
        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)
        length = end_line - start_line + 1
        if length > 50:
            self.findings.append(
                Finding(
                    tool="ast",
                    rule_id="long-function",
                    category="complexity",
                    severity="medium",
                    message=f"Function '{node.name}' is too long ({length} lines > 50 lines).",
                    line=start_line,
                    end_line=end_line,
                    confidence=0.85,
                    extra={"length": length, "function": node.name},
                )
            )

        # 2. Too many arguments (>6)
        all_args = (
            len(node.args.posonlyargs)
            + len(node.args.args)
            + len(node.args.kwonlyargs)
            + (1 if node.args.vararg else 0)
            + (1 if node.args.kwarg else 0)
        )
        if all_args > 6:
            self.findings.append(
                Finding(
                    tool="ast",
                    rule_id="too-many-args",
                    category="complexity",
                    severity="low",
                    message=f"Function '{node.name}' has {all_args} arguments (> 6).",
                    line=start_line,
                    end_line=getattr(node.args, "end_lineno", start_line),
                    confidence=0.9,
                    extra={"arg_count": all_args, "function": node.name},
                )
            )

        # 3. Mutable default arguments
        self._check_mutable_defaults(node)

        # 4. Cyclomatic complexity (>10 medium, >15 high)
        self._check_cyclomatic_complexity(node)

    def _check_mutable_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        all_defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
        for default in all_defaults:
            is_mutable = False
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                is_mutable = True
            elif isinstance(default, ast.Call):
                if isinstance(default.func, ast.Name) and default.func.id in {"list", "dict", "set"}:
                    is_mutable = True

            if is_mutable:
                self.findings.append(
                    Finding(
                        tool="ast",
                        rule_id="mutable-default-arg",
                        category="bug",
                        severity="medium",
                        message=f"Function '{node.name}' uses a mutable default argument. State will persist across calls.",
                        line=getattr(default, "lineno", node.lineno),
                        end_line=getattr(default, "end_lineno", getattr(default, "lineno", node.lineno)),
                        confidence=0.95,
                        extra={"function": node.name},
                    )
                )

    def _check_cyclomatic_complexity(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # Calculate McCabe Cyclomatic Complexity scoped to this function body (ignoring nested functions)
        complexity = 1

        for child in ast.walk(node):
            # Don't count constructs from inner nested functions toward this function's complexity
            if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if isinstance(child, (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif hasattr(ast, "match_case") and isinstance(child, ast.match_case):
                complexity += 1

        start_line = getattr(node, "lineno", 1)
        end_line = getattr(node, "end_lineno", start_line)

        if complexity > 15:
            self.findings.append(
                Finding(
                    tool="ast",
                    rule_id="cyclomatic-complexity-high",
                    category="complexity",
                    severity="high",
                    message=f"Function '{node.name}' has high cyclomatic complexity ({complexity} > 15). Consider refactoring.",
                    line=start_line,
                    end_line=end_line,
                    confidence=0.9,
                    extra={"complexity": complexity, "function": node.name},
                )
            )
        elif complexity > 10:
            self.findings.append(
                Finding(
                    tool="ast",
                    rule_id="cyclomatic-complexity-medium",
                    category="complexity",
                    severity="medium",
                    message=f"Function '{node.name}' has elevated cyclomatic complexity ({complexity} > 10).",
                    line=start_line,
                    end_line=end_line,
                    confidence=0.9,
                    extra={"complexity": complexity, "function": node.name},
                )
            )

    # 5. Bare except detection
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        is_bare = False
        if node.type is None:
            is_bare = True
        elif isinstance(node.type, ast.Name) and node.type.id == "BaseException":
            is_bare = True

        if is_bare:
            self.findings.append(
                Finding(
                    tool="ast",
                    rule_id="bare-except",
                    category="bug",
                    severity="medium",
                    message="Bare 'except:' or 'except BaseException:' catches system interrupts. Catch specific exceptions.",
                    line=getattr(node, "lineno", 1),
                    end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                    confidence=0.95,
                )
            )
        self.generic_visit(node)

    # 6. eval/exec detection
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            self.findings.append(
                Finding(
                    tool="ast",
                    rule_id="eval-exec",
                    category="security",
                    severity="high",
                    message=f"Dynamic code execution via '{node.func.id}()' is hazardous and allows arbitrary code execution.",
                    line=getattr(node, "lineno", 1),
                    end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                    confidence=0.95,
                    extra={"target": node.func.id},
                )
            )
        self.generic_visit(node)

    # 7. Comparison to None (== None, != None)
    def visit_Compare(self, node: ast.Compare) -> None:
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            is_eq_none = isinstance(op, (ast.Eq, ast.NotEq)) and (
                (isinstance(right, ast.Constant) and right.value is None)
                or (isinstance(left, ast.Constant) and left.value is None)
            )
            if is_eq_none:
                op_name = "==" if isinstance(op, ast.Eq) else "!="
                pref = "is" if isinstance(op, ast.Eq) else "is not"
                self.findings.append(
                    Finding(
                        tool="ast",
                        rule_id="comparison-to-none",
                        category="style",
                        severity="low",
                        message=f"Comparison with None using '{op_name}'. Use '{pref} None' instead.",
                        line=getattr(node, "lineno", 1),
                        end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                        confidence=0.9,
                    )
                )
            left = right
        self.generic_visit(node)

    # 8. Deep nesting (>4 levels)
    def _enter_nesting(self, node: ast.AST) -> None:
        self._nesting_depth += 1
        if self._nesting_depth > 4 and id(node) not in self._reported_nesting_nodes:
            self._reported_nesting_nodes.add(id(node))
            self.findings.append(
                Finding(
                    tool="ast",
                    rule_id="deep-nesting",
                    category="complexity",
                    severity="medium",
                    message=f"Deeply nested block detected (nesting depth {self._nesting_depth} > 4).",
                    line=getattr(node, "lineno", 1),
                    end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                    confidence=0.85,
                    extra={"depth": self._nesting_depth},
                )
            )

    def _exit_nesting(self) -> None:
        self._nesting_depth = max(0, self._nesting_depth - 1)

    def visit_If(self, node: ast.If) -> None:
        self._enter_nesting(node)
        self.generic_visit(node)
        self._exit_nesting()

    def visit_For(self, node: ast.For) -> None:
        self._enter_nesting(node)
        self.generic_visit(node)
        self._exit_nesting()

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._enter_nesting(node)
        self.generic_visit(node)
        self._exit_nesting()

    def visit_While(self, node: ast.While) -> None:
        self._enter_nesting(node)
        self.generic_visit(node)
        self._exit_nesting()

    def visit_Try(self, node: ast.Try) -> None:
        self._enter_nesting(node)
        self.generic_visit(node)
        self._exit_nesting()

    def visit_With(self, node: ast.With) -> None:
        self._enter_nesting(node)
        self.generic_visit(node)
        self._exit_nesting()

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._enter_nesting(node)
        self.generic_visit(node)
        self._exit_nesting()


def analyze_ast(code: str, filename: str = "<string>") -> List[Finding]:
    """Parse Python code and return all AST findings. Never crashes."""
    if not code or not code.strip():
        return []

    tree = None
    try:
        tree = ast.parse(code, filename=filename)
    except SyntaxError as exc:
        dedented = textwrap.dedent(code)
        if dedented != code:
            try:
                tree = ast.parse(dedented, filename=filename)
            except SyntaxError:
                tree = None
        if tree is None:
            lineno = exc.lineno or 1
            end_lineno = exc.end_lineno or lineno
            return [
                Finding(
                    tool="ast",
                    rule_id="syntax-error",
                    category="bug",
                    severity="high",
                    message=f"SyntaxError: {exc.msg}",
                    line=lineno,
                    end_line=end_lineno,
                    confidence=1.0,
                    extra={"offset": exc.offset, "text": exc.text or ""},
                )
            ]
    except Exception as exc:
        return [
            Finding(
                tool="ast",
                rule_id="parse-error",
                category="bug",
                severity="high",
                message=f"Parse failure: {exc}",
                line=1,
                end_line=1,
                confidence=1.0,
            )
        ]

    analyzer = ASTAnalyzer(filename=filename)
    try:
        analyzer.visit(tree)
    except Exception as exc:
        analyzer.findings.append(
            Finding(
                tool="ast",
                rule_id="analyzer-error",
                category="bug",
                severity="medium",
                message=f"Error during AST traversal: {exc}",
                line=1,
                end_line=1,
                confidence=0.5,
            )
        )

    return analyzer.findings
