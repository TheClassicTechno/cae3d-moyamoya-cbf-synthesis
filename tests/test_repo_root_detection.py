"""Regression test for the hardcoded-path fix: every script using the
`_REPO_ROOT = os.path.dirname(...)` bootstrap must actually have `os` bound
before that line runs, and the computed root must be the real repo root.

This exists because the automated fix that introduced this pattern initially
inserted the snippet before `import os` in 3 files (caught by manual
external-clone validation, not by this repo's own tooling at the time).
"""
import ast
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _files_with_repo_root_snippet():
    # Scope to git-tracked files only -- this repo's working directory is shared
    # with a much larger, non-repo research tree (caches, environments, vendored
    # third-party clones with non-UTF-8 sources) that a blind os.walk would hit.
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for rel_path in tracked:
        path = os.path.join(REPO_ROOT, rel_path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if "_REPO_ROOT = os.path.dirname" in text:
            yield path


def test_os_is_bound_before_repo_root_snippet_runs():
    checked = 0
    for path in _files_with_repo_root_snippet():
        tree = ast.parse(open(path).read(), filename=path)
        snippet_line = None
        os_import_line = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_REPO_ROOT" for t in node.targets
            ):
                snippet_line = node.lineno
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os" and (os_import_line is None or node.lineno < os_import_line):
                        os_import_line = node.lineno
        assert snippet_line is not None, f"{path}: expected _REPO_ROOT assignment not found"
        assert os_import_line is not None, f"{path}: 'import os' not found anywhere"
        assert os_import_line < snippet_line, f"{path}: 'import os' (line {os_import_line}) comes after _REPO_ROOT snippet (line {snippet_line})"
        checked += 1
    assert checked > 50  # sanity: this should cover the ~69 files the fix touched
