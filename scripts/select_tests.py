
import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
TEST_DIR = ROOT / "tests"


def get_changed_files():
    """Find source files changed in the latest commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True
        )

        changed = set(result.stdout.splitlines())

        return {
            Path(file).stem
            for file in changed
            if file.startswith("src/") and file.endswith(".py")
        }

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_imports(file_path):
    """Extract local source-module imports from a Python file."""
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return set()

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")

                if parts[0] == "src" and len(parts) > 1:
                    imports.add(parts[1])
                else:
                    imports.add(parts[0])

        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")

            if parts[0] == "src" and len(parts) > 1:
                imports.add(parts[1])
            else:
                imports.add(parts[0])

    return imports


def select_tests():
    changed = get_changed_files()

    all_tests = sorted(TEST_DIR.glob("test_*.py"))

    # If Git diff fails or no source files changed, run everything.
    if changed is None or not changed:
        return all_tests

    # Build the source dependency graph.
    dependencies = {}

    for source_file in SRC_DIR.glob("*.py"):
        dependencies[source_file.stem] = get_imports(source_file)

    # Expand changed files to include dependent source modules.
    impact = set(changed)

    updated = True
    while updated:
        updated = False

        for module, imports in dependencies.items():
            if module not in impact and imports.intersection(impact):
                impact.add(module)
                updated = True

    # Select tests that import any affected source module.
    selected = []

    for test_file in all_tests:
        test_imports = get_imports(test_file)

        if test_imports.intersection(impact):
            selected.append(test_file)

    # Safety fallback: never run zero tests.
    return selected if selected else all_tests


if __name__ == "__main__":
    tests = select_tests()

    print(
        " ".join(
            str(test.relative_to(ROOT)).replace("\\", "/")
            for test in tests
        )
    )
