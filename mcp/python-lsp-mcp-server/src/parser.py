# parser.py
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SymbolDefinition:
    """Description of a symbol definition in the codebase.

    Parameters
    ----------
    parent_qualified_name : str or None
        Fully qualified name of the parent symbol. For module-level symbols
        this is typically the module's qualified name, and for modules it is
        ``None``.
    name : str
        Simple (unqualified) name of the symbol, e.g. ``"MyClass"``.
    kind : str
        Kind of symbol. Common values are ``"module"``, ``"class"``,
        ``"function"``, and ``"variable"``.
    type_annotation : str or None
        Textual representation of the type annotation for the symbol, if
        available. For functions and classes, this is usually ``None``.
    file_path : pathlib.Path
        Absolute path to the file where the symbol is defined.
    line : int
        1-based line number of the symbol definition.
    column : int
        0-based column offset of the symbol definition.
    docstring : str or None
        Docstring associated with the symbol, if any.
    init_docstring : str or None
        Docstring associated with the ``__init__`` method if the symbol is a
        class and such a method exists.
    full_source : str or None
        Source code corresponding to the full definition of the symbol,
        including its body.
    qualified_name : str
        Fully qualified name of the symbol, including module and parents,
        e.g. ``"package.module.MyClass.method"``.
    """

    parent_qualified_name: Optional[str]
    name: str
    kind: str
    type_annotation: Optional[str]
    file_path: Path
    line: int
    column: int
    docstring: Optional[str]
    init_docstring: Optional[str]
    full_source: Optional[str]
    qualified_name: str


@dataclass
class OutlineNode:
    """Node in the outline tree.

    Each node wraps a :class:`SymbolDefinition` and may have child nodes
    representing nested symbols (e.g., methods inside a class).

    Parameters
    ----------
    symbol : SymbolDefinition
        The symbol definition represented by this outline node.
    children : list of OutlineNode, optional
        Child nodes nested under this symbol. Defaults to an empty list.
    """

    symbol: SymbolDefinition
    children: List["OutlineNode"] = field(default_factory=list)


@dataclass
class SymbolReference:
    """Description of a symbol reference in the codebase.

    Parameters
    ----------
    symbol : str
        The name of the referenced symbol (simple name, not necessarily
        qualified).
    file_path : pathlib.Path
        Absolute path to the file where the reference occurs.
    line : int
        1-based line number where the reference appears.
    column : int
        0-based column offset where the reference appears.
    context : str or None, optional
        Optional context block, typically a window of lines around the
        reference location (for example, +/- 1 line, i.e. 3 lines total).
    """

    symbol: str
    file_path: Path
    line: int
    column: int
    context: Optional[str] = None


def _compute_module_qualified_name(
    file_path: Path,
    project_root: Optional[Path] = None,
) -> str:
    """Compute a module's qualified name from its file path.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the Python file.
    project_root : pathlib.Path or None, optional
        Root of the project. If provided and `file_path` is under this root,
        the qualified name is derived from the relative path. Otherwise,
        the module's stem is used.

    Returns
    -------
    str
        The computed module qualified name.
    """
    file_path = file_path.resolve()
    if project_root is not None:
        try:
            rel = file_path.relative_to(project_root.resolve())
            return ".".join(rel.with_suffix("").parts)
        except ValueError:
            # file not under project root
            pass
    return file_path.stem


def _extract_context_window(
    text: str,
    line: int,
    window: int = 1,
) -> str:
    """Extract a window of lines around a reference.

    Parameters
    ----------
    text : str
        Full source text of the file.
    line : int
        1-based line number of the reference.
    window : int, optional
        Number of lines to include above and below the reference. Default
        is 1 (i.e. 3 lines total, where possible).

    Returns
    -------
    str
        Context block as a single string with newline separators.
    """
    lines = text.splitlines()
    idx = max(0, line - 1)
    start = max(0, idx - window)
    end = min(len(lines), idx + window + 1)
    return "\n".join(lines[start:end])


def parse_python_file(
    file_path: Path,
    project_root: Optional[Path] = None,
) -> Tuple[OutlineNode, Dict[str, SymbolDefinition], List[SymbolReference]]:
    """Parse a Python file into an outline tree, symbol definitions, and references.

    This function builds:

    - a top-level :class:`OutlineNode` representing the module, with nested
      children for classes, functions, and variables;
    - a dictionary mapping fully qualified symbol names to their
      :class:`SymbolDefinition`;
    - a list of :class:`SymbolReference` objects for all Name occurrences.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the Python source file.
    project_root : pathlib.Path or None, optional
        Root of the project, used to compute module-qualified names. If
        omitted, the module name defaults to the file stem.

    Returns
    -------
    module_node : OutlineNode
        The root outline node representing the module.
    definitions : dict of str to SymbolDefinition
        Mapping from qualified symbol names to their definitions.
    references : list of SymbolReference
        List of symbol references found in the file.
    """
    file_path = file_path.resolve()
    text = file_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(file_path))

    module_qualname = _compute_module_qualified_name(file_path, project_root)
    module_doc = ast.get_docstring(tree, clean=True)
    module_def = SymbolDefinition(
        parent_qualified_name=None,
        name=module_qualname.split(".")[-1],
        kind="module",
        type_annotation=None,
        file_path=file_path,
        line=1,
        column=0,
        docstring=module_doc,
        init_docstring=None,
        full_source=text,
        qualified_name=module_qualname,
    )
    module_node = OutlineNode(symbol=module_def)

    definitions: Dict[str, SymbolDefinition] = {module_def.qualified_name: module_def}

    def walk_body(
        body: List[ast.stmt],
        parent_def: SymbolDefinition,
        parent_node: OutlineNode,
    ) -> None:
        """Recursively walk AST body to populate definitions and outline.

        Parameters
        ----------
        body : list of ast.stmt
            Statements in the current scope.
        parent_def : SymbolDefinition
            Definition of the current enclosing symbol (module, class, or
            function).
        parent_node : OutlineNode
            Outline node corresponding to `parent_def`.
        """
        for node in body:
            # Classes
            if isinstance(node, ast.ClassDef):
                qualname = f"{parent_def.qualified_name}.{node.name}"
                doc = ast.get_docstring(node, clean=True)
                init_doc = None
                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == "__init__"
                    ):
                        init_doc = ast.get_docstring(child, clean=True)
                        break
                full_src = ast.get_source_segment(text, node) or ""
                cls_def = SymbolDefinition(
                    parent_qualified_name=parent_def.qualified_name,
                    name=node.name,
                    kind="class",
                    type_annotation=None,
                    file_path=file_path,
                    line=getattr(node, "lineno", 0),
                    column=getattr(node, "col_offset", 0),
                    docstring=doc,
                    init_docstring=init_doc,
                    full_source=full_src,
                    qualified_name=qualname,
                )
                definitions[cls_def.qualified_name] = cls_def
                cls_node = OutlineNode(symbol=cls_def)
                parent_node.children.append(cls_node)
                walk_body(node.body, cls_def, cls_node)

            # Functions (sync + async)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{parent_def.qualified_name}.{node.name}"
                doc = ast.get_docstring(node, clean=True)
                full_src = ast.get_source_segment(text, node) or ""
                fn_def = SymbolDefinition(
                    parent_qualified_name=parent_def.qualified_name,
                    name=node.name,
                    kind="function",
                    type_annotation=None,
                    file_path=file_path,
                    line=getattr(node, "lineno", 0),
                    column=getattr(node, "col_offset", 0),
                    docstring=doc,
                    init_docstring=None,
                    full_source=full_src,
                    qualified_name=qualname,
                )
                definitions[fn_def.qualified_name] = fn_def
                fn_node = OutlineNode(symbol=fn_def)
                parent_node.children.append(fn_node)
                walk_body(node.body, fn_def, fn_node)

            # Simple variables
            elif isinstance(node, ast.Assign):
                full_src = ast.get_source_segment(text, node) or ""
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        qualname = f"{parent_def.qualified_name}.{target.id}"
                        var_def = SymbolDefinition(
                            parent_qualified_name=parent_def.qualified_name,
                            name=target.id,
                            kind="variable",
                            type_annotation=None,
                            file_path=file_path,
                            line=getattr(node, "lineno", 0),
                            column=getattr(node, "col_offset", 0),
                            docstring=None,
                            init_docstring=None,
                            full_source=full_src,
                            qualified_name=qualname,
                        )
                        definitions[var_def.qualified_name] = var_def
                        var_node = OutlineNode(symbol=var_def)
                        parent_node.children.append(var_node)

            # Annotated assignment
            elif isinstance(node, ast.AnnAssign):
                full_src = ast.get_source_segment(text, node) or ""
                type_ann = None
                if node.annotation is not None:
                    try:
                        type_ann = ast.unparse(node.annotation)  # Python 3.9+
                    except AttributeError:
                        type_ann = None
                if isinstance(node.target, ast.Name):
                    qualname = f"{parent_def.qualified_name}.{node.target.id}"
                    var_def = SymbolDefinition(
                        parent_qualified_name=parent_def.qualified_name,
                        name=node.target.id,
                        kind="variable",
                        type_annotation=type_ann,
                        file_path=file_path,
                        line=getattr(node, "lineno", 0),
                        column=getattr(node, "col_offset", 0),
                        docstring=None,
                        init_docstring=None,
                        full_source=full_src,
                        qualified_name=qualname,
                    )
                    definitions[var_def.qualified_name] = var_def
                    var_node = OutlineNode(symbol=var_def)
                    parent_node.children.append(var_node)

            # You can extend here to handle imports, constants, etc.

    walk_body(tree.body, module_def, module_node)

    # Collect references via a single tree walk.
    references: List[SymbolReference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            line = getattr(node, "lineno", 0)
            col = getattr(node, "col_offset", 0)
            context = _extract_context_window(text, line, window=1)
            references.append(
                SymbolReference(
                    symbol=node.id,
                    file_path=file_path,
                    line=line,
                    column=col,
                    context=context,
                )
            )

    return module_node, definitions, references
