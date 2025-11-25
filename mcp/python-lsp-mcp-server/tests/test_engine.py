# tests/test_engine.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from engine import LSPEngine  # type: ignore

BASIC_SOURCE = """\
class MyClass:
    \"\"\"This is MyClass docstring.\"\"\"

    def __init__(self, x: int):
        \"\"\"Initialises MyClass with x.\"\"\"
        self.x = x

    def method(self, y: int) -> int:
        \"\"\"Does something with y.\"\"\"
        return self.x + y


def my_function(a: int, b: int) -> int:
    \"\"\"This is my_function docstring.\"\"\"
    return a * b


obj = MyClass(10)
res = obj.method(5)
out = my_function(3, 4)
"""


OTHER_SOURCE = """\
class MyClass:
    \"\"\"MyClass in another file.\"\"\"
    pass
"""


COLLISION_SOURCE_A = """\
def util():
    \"\"\"util in A\"\"\"
    return 1
"""

COLLISION_SOURCE_B = """\
def util():
    \"\"\"util in B\"\"\"
    return 2
"""


@pytest.fixture
def small_project(tmp_path: Path) -> Dict[str, Path]:
    """Create a small project with a single demo file."""
    file_path = tmp_path / "demo.py"
    file_path.write_text(BASIC_SOURCE, encoding="utf-8")
    return {"root": tmp_path, "demo": file_path}


@pytest.fixture
def multi_file_project(tmp_path: Path) -> Dict[str, Path]:
    """Project with multiple files, including collisions."""
    main_path = tmp_path / "main.py"
    other_path = tmp_path / "other.py"
    a_path = tmp_path / "pkg_a.py"
    b_path = tmp_path / "pkg_b.py"

    main_path.write_text(BASIC_SOURCE, encoding="utf-8")
    other_path.write_text(OTHER_SOURCE, encoding="utf-8")
    a_path.write_text(COLLISION_SOURCE_A, encoding="utf-8")
    b_path.write_text(COLLISION_SOURCE_B, encoding="utf-8")

    return {
        "root": tmp_path,
        "main": main_path,
        "other": other_path,
        "a": a_path,
        "b": b_path,
    }


def test_initial_index_eager_small_project(small_project: Dict[str, Path]) -> None:
    """Engine should eagerly index a small project and serve basic queries."""
    root = small_project["root"]
    demo_file = small_project["demo"]

    engine = LSPEngine(
        project_root=root,
        max_eager_files=10,
        max_eager_bytes=10_000,
    )

    # Symbol-based definition short
    short = engine.get_definition_short(symbol="MyClass")
    assert short is not None
    assert short["name"] == "MyClass"
    assert "docstring" in short
    assert "MyClass docstring" in (short["docstring"] or "")

    # File-based definition short (module)
    module_short = engine.get_definition_short(file_path=str(demo_file))
    assert module_short is not None
    assert module_short["kind"] == "module"

    # Full definition
    full = engine.get_definition_full(symbol="MyClass")
    assert full is not None
    assert "source" in full
    assert "class MyClass" in full["source"]

    # Outline by file
    outline = engine.get_outline(file_path=str(demo_file))
    names = {n["name"] for n in outline}
    assert "MyClass" in names
    assert "my_function" in names

    # References by symbol
    refs = engine.get_references(symbol="MyClass")
    assert refs
    assert any(r["file_path"].endswith("demo.py") for r in refs)


def test_lazy_indexing_for_symbol(multi_file_project: Dict[str, Path]) -> None:
    """Engine should lazily index files when a symbol is requested."""
    root = multi_file_project["root"]
    main_file = multi_file_project["main"]

    # Force extremely small eager limits to ensure lazy behavior.
    engine = LSPEngine(
        project_root=root,
        max_eager_files=0,
        max_eager_bytes=0,
    )

    # At this point, nothing should be indexed eagerly.
    # Requesting MyClass should trigger indexing of relevant files.
    short = engine.get_definition_short(symbol="MyClass")
    assert short is not None
    assert short["name"] == "MyClass"
    # It should be in either main.py or other.py
    assert short["file_path"].endswith((".py",))


def test_truncation_in_definition_short(small_project: Dict[str, Path]) -> None:
    """Definition short should respect max_chars for docstrings."""
    root = small_project["root"]
    engine = LSPEngine(project_root=root, max_eager_files=10, max_eager_bytes=10_000)

    short = engine.get_definition_short(symbol="MyClass", max_chars=10)
    assert short is not None
    doc = short["docstring"]
    assert doc is not None
    assert len(doc) <= 11  # 10 chars + ellipsis


def test_truncation_in_definition_full(small_project: Dict[str, Path]) -> None:
    """Definition full should respect max_chars for source."""
    root = small_project["root"]
    engine = LSPEngine(project_root=root, max_eager_files=10, max_eager_bytes=10_000)

    full = engine.get_definition_full(symbol="MyClass", max_chars=40)
    assert full is not None
    src = full["source"]
    assert len(src) <= 40 + len("\n…(truncated)")


def test_outline_by_symbol(small_project: Dict[str, Path]) -> None:
    """Outline should work from a symbol node downward."""
    root = small_project["root"]
    demo_file = small_project["demo"]
    engine = LSPEngine(project_root=root, max_eager_files=10, max_eager_bytes=10_000)

    # Get outline for module, then symbol
    module_outline = engine.get_outline(file_path=str(demo_file))
    myclass_node = next(n for n in module_outline if n["name"] == "MyClass")

    # Outline for MyClass symbol should show methods as children
    class_outline = engine.get_outline(symbol="MyClass")
    # first-level children of class
    child_names = {n["name"] for n in class_outline}
    assert "__init__" in child_names
    assert "method" in child_names


def test_references_filtered_by_file(small_project: Dict[str, Path]) -> None:
    """References should be filterable by file path."""
    root = small_project["root"]
    demo_file = small_project["demo"]
    engine = LSPEngine(project_root=root, max_eager_files=10, max_eager_bytes=10_000)

    refs = engine.get_references(file_path=str(demo_file))
    assert refs
    assert all(r["file_path"].endswith("demo.py") for r in refs)


def test_symbol_collisions_multi_file(multi_file_project: Dict[str, Path]) -> None:
    """When multiple files define the same symbol, engine should still return something sensible."""
    root = multi_file_project["root"]
    engine = LSPEngine(project_root=root, max_eager_files=10, max_eager_bytes=10_000)

    # Both pkg_a and pkg_b define util(); current strategy picks first qualname
    short = engine.get_definition_short(symbol="util")
    assert short is not None
    assert short["name"] == "util"
    # Implementation detail: which file we get is currently unspecified, but
    # we should get one of pkg_a.py or pkg_b.py.
    assert short["file_path"].endswith(("pkg_a.py", "pkg_b.py"))
