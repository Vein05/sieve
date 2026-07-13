"""Verify all internal imports resolve to existing modules/files.

Uses AST parsing to extract import statements from every .py file in the repo,
then checks that each internal import target actually exists as a file or package.
No runtime imports are performed, so this test works even when heavy dependencies
(spaCy models, GPU libs, API keys) are unavailable.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Top-level packages in the repo (directories with __init__.py or standalone modules)
_SOURCE_PACKAGES = {
    "cli", "compiler", "controller", "evidence", "reader",
    "retrieval", "shared", "analysis",
}


def _collect_py_files() -> list[Path]:
    """Walk the repo and collect all .py files under source packages."""
    py_files: list[Path] = []
    for pkg in _SOURCE_PACKAGES:
        pkg_dir = REPO_ROOT / pkg
        if not pkg_dir.is_dir():
            continue
        for root, _dirs, files in os.walk(pkg_dir):
            for fname in files:
                if fname.endswith(".py"):
                    py_files.append(Path(root) / fname)
    return py_files


def _extract_imports(filepath: Path) -> list[tuple[str, int]]:
    """Parse a .py file and return (dotted_module_path, line_number) for each import."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Resolve relative imports
                if node.level > 0:
                    # Relative import: compute the absolute module path
                    parts = filepath.relative_to(REPO_ROOT).parts
                    # Go up `level` directories from the file's package
                    pkg_parts = parts[: -(node.level)]
                    if pkg_parts:
                        full_module = ".".join(pkg_parts) + "." + node.module
                    else:
                        full_module = node.module
                else:
                    full_module = node.module
                imports.append((full_module, node.lineno))
    return imports


def _is_internal_import(module_path: str) -> bool:
    """Check if a dotted import path refers to one of our source packages."""
    top = module_path.split(".")[0]
    return top in _SOURCE_PACKAGES


def _module_path_exists(module_path: str) -> bool:
    """Check that a dotted module path resolves to a file or package on disk."""
    parts = module_path.split(".")
    # Try as a package (directory with __init__.py)
    candidate_dir = REPO_ROOT / Path(*parts)
    if candidate_dir.is_dir() and (candidate_dir / "__init__.py").exists():
        return True
    # Try as a module file
    candidate_file = REPO_ROOT / Path(*parts[:-1]) / (parts[-1] + ".py") if len(parts) > 1 else REPO_ROOT / (parts[0] + ".py")
    if candidate_file.is_file():
        return True
    # The import might be importing a name FROM a module (e.g., from compiler.types import X)
    # In that case, the module is parts[:-1]
    if len(parts) >= 2:
        parent_module = ".".join(parts[:-1])
        return _module_path_exists(parent_module)
    return False


def test_all_internal_imports_resolve():
    """Verify no broken import paths after restructure."""
    py_files = _collect_py_files()
    assert len(py_files) > 0, "No Python files found in source packages"

    broken: list[str] = []
    for filepath in py_files:
        for module_path, lineno in _extract_imports(filepath):
            if not _is_internal_import(module_path):
                continue
            if not _module_path_exists(module_path):
                rel = filepath.relative_to(REPO_ROOT)
                broken.append(f"  {rel}:{lineno} -> {module_path}")

    if broken:
        msg = f"Found {len(broken)} broken internal import(s):\n" + "\n".join(broken)
        pytest.fail(msg)


def test_source_packages_have_init():
    """Every source package directory must have an __init__.py."""
    missing: list[str] = []
    for pkg in _SOURCE_PACKAGES:
        pkg_dir = REPO_ROOT / pkg
        if pkg_dir.is_dir() and not (pkg_dir / "__init__.py").exists():
            missing.append(pkg)
    if missing:
        pytest.fail(f"Missing __init__.py in packages: {missing}")


def test_no_wildcard_imports():
    """CLAUDE.md forbids wildcard imports; verify none exist."""
    py_files = _collect_py_files()
    violations: list[str] = []
    for filepath in py_files:
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(filepath))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        rel = filepath.relative_to(REPO_ROOT)
                        violations.append(f"  {rel}:{node.lineno} -> from {node.module} import *")
    if violations:
        pytest.fail(f"Wildcard imports found:\n" + "\n".join(violations))
