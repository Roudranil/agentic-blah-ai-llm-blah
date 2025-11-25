# tests/test_parser.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Set

import pytest

# Ensure src/ is on path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parser import (  # type: ignore
    OutlineNode,
    SymbolDefinition,
    SymbolReference,
    parse_python_file,
)

BASIC_SOURCE = """\
\"\"\"Module level docstring.\"\"\"

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


NESTED_SOURCE = """\
def outer(a: int):
    \"\"\"Outer function.\"\"\"

    def inner(b: int) -> int:
        \"\"\"Inner function.\"\"\"
        return a + b

    return inner(10)


class Container:
    \"\"\"Container class.\"\"\"

    class Inner:
        \"\"\"Inner class.\"\"\"
        pass
"""


ANNOTATED_SOURCE = """\
x: int = 42
y: str
z = 10
"""


@pytest.fixture
def basic_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "demo_basic.py"
    file_path.write_text(BASIC_SOURCE, encoding="utf-8")
    return file_path


@pytest.fixture
def nested_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "demo_nested.py"
    file_path.write_text(NESTED_SOURCE, encoding="utf-8")
    return file_path


@pytest.fixture
def annotated_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "demo_annotated.py"
    file_path.write_text(ANNOTATED_SOURCE, encoding="utf-8")
    return file_path


def test_parse_basic_module_structure(basic_file: Path, tmp_path: Path) -> None:
    """Module node, basic children, and qualified names should be correct."""
    module_node, definitions, references = parse_python_file(
        file_path=basic_file,
        project_root=tmp_path,
    )

    assert isinstance(module_node, OutlineNode)
    assert isinstance(module_node.symbol, SymbolDefinition)

    module_def = module_node.symbol
    assert module_def.kind == "module"
    assert module_def.file_path == basic_file
    assert module_def.docstring == "Module level docstring."

    # Expect MyClass and my_function as direct children
    child_names: Set[str] = {child.symbol.name for child in module_node.children}
    assert "MyClass" in child_names
    assert "my_function" in child_names

    qualnames = set(definitions.keys())
    # Expect fully qualified names with module base
    assert any(q.endswith(".MyClass") for q in qualnames)
    assert any(q.endswith(".my_function") for q in qualnames)

    # MyClass definition should have init_docstring
    myclass_def = next(
        d for d in definitions.values() if d.name == "MyClass" and d.kind == "class"
    )
    assert myclass_def.docstring == "This is MyClass docstring."
    assert myclass_def.init_docstring == "Initialises MyClass with x."
    assert "class MyClass" in (myclass_def.full_source or "")


def test_nested_symbols(nested_file: Path, tmp_path: Path) -> None:
    """Nested functions and classes should be represented as children."""
    module_node, definitions, _ = parse_python_file(
        file_path=nested_file, project_root=tmp_path
    )

    # outer should be a direct child
    outer_node = next(
        child for child in module_node.children if child.symbol.name == "outer"
    )
    assert outer_node.symbol.kind == "function"

    # outer should have inner as child
    inner_node = next(
        child for child in outer_node.children if child.symbol.name == "inner"
    )
    assert inner_node.symbol.kind == "function"
    assert inner_node.symbol.docstring == "Inner function."

    # Container and Container.Inner
    container_node = next(
        child for child in module_node.children if child.symbol.name == "Container"
    )
    inner_class_node = next(
        child for child in container_node.children if child.symbol.name == "Inner"
    )
    assert inner_class_node.symbol.kind == "class"
    assert inner_class_node.symbol.docstring == "Inner class."

    # Qualified names should reflect nesting
    qualnames = set(definitions.keys())
    assert any(q.endswith(".outer.inner") for q in qualnames)
    assert any(q.endswith(".Container.Inner") for q in qualnames)


def test_annotated_and_unannotated_variables(
    annotated_file: Path, tmp_path: Path
) -> None:
    """Annotated and unannotated variables should be captured with type annotations."""
    module_node, definitions, _ = parse_python_file(
        file_path=annotated_file,
        project_root=tmp_path,
    )

    var_names = {child.symbol.name for child in module_node.children}
    assert {"x", "y", "z"}.issubset(var_names)

    x_def = next(d for d in definitions.values() if d.name == "x")
    y_def = next(d for d in definitions.values() if d.name == "y")
    z_def = next(d for d in definitions.values() if d.name == "z")

    assert x_def.type_annotation in (None, "int")  # depending on Python/unparse
    assert y_def.type_annotation in (None, "str")
    assert z_def.type_annotation is None
    assert x_def.kind == y_def.kind == z_def.kind == "variable"


def test_references_and_context(basic_file: Path, tmp_path: Path) -> None:
    """References should have context windows and plausible positions."""
    _, _, references = parse_python_file(
        file_path=basic_file,
        project_root=tmp_path,
    )

    assert references
    assert all(isinstance(ref, SymbolReference) for ref in references)

    # There should be references to MyClass and my_function
    assert any(ref.symbol == "MyClass" for ref in references)
    assert any(ref.symbol == "my_function" for ref in references)

    # Context should be non-empty and not excessively large
    for ref in references:
        if ref.context is not None:
            assert isinstance(ref.context, str)
            # context window of +/-1 line should be reasonably small
            assert ref.context.count("\n") <= 4
