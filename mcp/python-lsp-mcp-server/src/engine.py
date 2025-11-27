# engine.py
from __future__ import annotations

from parser import OutlineNode, SymbolDefinition, SymbolReference, parse_python_file
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

MAX_CHARS_HARD_LIMIT = 1500


class LSPEngine:
    """Engine for indexing Python projects and serving code intelligence queries.

    The engine builds and maintains a sparse index of a project, using
    :class:`OutlineNode` trees and associated :class:`SymbolDefinition` and
    :class:`SymbolReference` objects. It is designed to scale to large
    codebases by:

    - discovering files and estimating project size;
    - eagerly indexing only a configurable subset of small files;
    - lazily indexing additional files on demand when queries require them.

    Parameters
    ----------
    project_root : str or pathlib.Path
        Root directory of the Python project to index.
    max_eager_files : int, optional
        Maximum number of files to index eagerly at initialization. Files
        are chosen starting from the smallest by size. Default is 200.
    max_eager_bytes : int, optional
        Maximum total byte size to eagerly index at initialization. Default
        is 5,000,000 (approximately 5 MB).

    Attributes
    ----------
    project_root : pathlib.Path
        Absolute path to the project root.
    outline_index : dict
        Mapping from file paths to root :class:`OutlineNode` objects for
        each indexed file (the module nodes).
    symbol_index : dict
        Mapping from fully qualified symbol names to :class:`SymbolDefinition`.
    references : list of SymbolReference
        List of all symbol references that have been collected so far.
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        max_eager_files: int = 200,
        max_eager_bytes: int = 5_000_000,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.max_eager_files = max_eager_files
        self.max_eager_bytes = max_eager_bytes

        # File-level outline trees.
        self.outline_index: Dict[Path, OutlineNode] = {}

        # Symbol definitions.
        # Qualified name -> definition.
        self.symbol_index: Dict[str, SymbolDefinition] = {}
        # Simple name -> list of qualified names (for convenience lookups).
        self._symbols_by_simple_name: Dict[str, List[str]] = {}

        # Symbol references.
        # Full list as requested.
        self.references: List[SymbolReference] = []
        # By symbol name (simple).
        self._refs_by_symbol: Dict[str, List[SymbolReference]] = {}

        # Outline nodes indexed by qualified name for outline/tree lookups.
        self._node_index: Dict[str, OutlineNode] = {}

        # File discovery & indexing state.
        self._all_files: List[Path] = []
        self._file_sizes: Dict[Path, int] = {}
        self._total_bytes: int = 0
        self._indexed_files: set[Path] = set()

        self._discover_files()
        self._initial_index()

    # ------------------------------------------------------------------
    # Internal indexing helpers
    # ------------------------------------------------------------------
    def _discover_files(self) -> None:
        """Discover Python files under the project root.

        This method populates the list of candidate files and approximate
        total project size. It does not parse any file contents.

        Notes
        -----
        This function is intentionally lightweight so that it can be run
        even on large projects without significant overhead.
        """
        total_bytes = 0
        all_files: List[Path] = []
        file_sizes: Dict[Path, int] = {}

        for path in self.project_root.rglob("*.py"):
            try:
                stat = path.stat()
                size = int(stat.st_size)
            except OSError:
                size = 0
            path = path.resolve()
            all_files.append(path)
            file_sizes[path] = size
            total_bytes += size

        self._all_files = all_files
        self._file_sizes = file_sizes
        self._total_bytes = total_bytes

    def _initial_index(self) -> None:
        """Perform an initial sparse index of the project.

        Notes
        -----
        If the project is small enough (under both file-count and size
        thresholds), all files are indexed eagerly. Otherwise, only the
        smallest files (by size) are indexed up to the configured limits.
        Additional files are indexed lazily when needed.
        """
        if not self._all_files:
            return

        files_sorted = sorted(self._all_files, key=lambda p: self._file_sizes.get(p, 0))

        if (
            len(self._all_files) <= self.max_eager_files
            and self._total_bytes <= self.max_eager_bytes
        ):
            to_index = files_sorted
        else:
            to_index = files_sorted[: self.max_eager_files]

        for file_path in to_index:
            self._index_file(file_path)

    def _index_file(self, file_path: Path) -> None:
        """Index a single file if it has not been indexed yet.

        Parameters
        ----------
        file_path : pathlib.Path
            Path to the Python file.

        Notes
        -----
        This function parses the file, builds its outline tree, collects
        symbol definitions and references, and updates internal indexes.
        """
        file_path = file_path.resolve()
        if file_path in self._indexed_files:
            return

        if not file_path.exists():
            return

        module_node, definitions, references = parse_python_file(
            file_path=file_path, project_root=self.project_root
        )

        self.outline_index[file_path] = module_node
        self._indexed_files.add(file_path)

        # Register outline nodes (module_node and its children).
        self._register_outline_node(module_node)

        # Register definitions.
        for defn in definitions.values():
            self._register_definition(defn)

        # Register references.
        for ref in references:
            self._register_reference(ref)

    def _register_outline_node(self, node: OutlineNode) -> None:
        """Register an outline node and its children in the node index.

        Parameters
        ----------
        node : OutlineNode
            Node to be registered.
        """
        qualname = node.symbol.qualified_name
        self._node_index[qualname] = node
        for child in node.children:
            self._register_outline_node(child)

    def _register_definition(self, defn: SymbolDefinition) -> None:
        """Register a symbol definition in the symbol indexes.

        Parameters
        ----------
        defn : SymbolDefinition
            The symbol definition to register.
        """
        self.symbol_index[defn.qualified_name] = defn
        self._symbols_by_simple_name.setdefault(defn.name, []).append(
            defn.qualified_name
        )

    def _register_reference(self, ref: SymbolReference) -> None:
        """Register a symbol reference in the reference indexes.

        Parameters
        ----------
        ref : SymbolReference
            The symbol reference to register.
        """
        self.references.append(ref)
        self._refs_by_symbol.setdefault(ref.symbol, []).append(ref)

    def _ensure_file_indexed(self, file_path: Union[str, Path]) -> None:
        """Ensure that a given file has been indexed, indexing it if needed.

        Parameters
        ----------
        file_path : str or pathlib.Path
            Path to the Python file.
        """
        path = Path(file_path).resolve()
        if path not in self._indexed_files:
            # If this file was not discovered initially, add it now.
            if path not in self._all_files:
                self._all_files.append(path)
                try:
                    self._file_sizes[path] = int(path.stat().st_size)
                except OSError:
                    self._file_sizes[path] = 0
            self._index_file(path)

    def _ensure_symbol_indexed(self, symbol: str) -> None:
        """Ensure that a given symbol has been indexed.

        This method attempts to find a definition for the symbol by
        progressively indexing additional files, prioritizing files whose
        names look related to the symbol.

        Parameters
        ----------
        symbol : str
            Simple or qualified symbol name.
        """
        if symbol in self.symbol_index or symbol in self._symbols_by_simple_name:
            return

        remaining = [p for p in self._all_files if p not in self._indexed_files]
        if not remaining:
            return

        # Prioritize files whose stem contains the symbol, then by size.
        def score(path: Path) -> Tuple[int, int]:
            stem = path.stem.lower()
            match = 0 if symbol.lower() in stem else 1
            size = self._file_sizes.get(path, 0)
            return (match, size)

        for path in sorted(remaining, key=score):
            self._index_file(path)
            if symbol in self.symbol_index or symbol in self._symbols_by_simple_name:
                break

    def _resolve_symbol_qualified_name(self, symbol: str) -> Optional[str]:
        """Resolve a symbol name to a qualified name if possible.

        Parameters
        ----------
        symbol : str
            Symbol name, either simple or fully qualified.

        Returns
        -------
        str or None
            Qualified symbol name if found, otherwise ``None``.
        """
        if symbol in self.symbol_index:
            return symbol
        if symbol in self._symbols_by_simple_name:
            # Choose the first occurrence; you may want to refine this.
            return self._symbols_by_simple_name[symbol][0]
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_definition_short(
        self,
        symbol: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        max_chars: int = 1000,
    ) -> Optional[Dict[str, Any]]:
        """Return a short definition for a symbol or module.

        Parameters
        ----------
        symbol : str, optional
            Symbol name or qualified name for which to retrieve the
            definition. If provided, this takes precedence over `file_path`.
        file_path : str or pathlib.Path, optional
            Path to a Python file. If provided and `symbol` is ``None``,
            the module-level definition is returned.
        max_chars : int, optional
            Maximum length for textual fields such as docstrings. Longer
            text is truncated with an ellipsis. Default is 1000.

        Returns
        -------
        dict or None
            Dictionary containing a compact description of the symbol, or
            ``None`` if the symbol cannot be found.
        """
        max_chars = min(MAX_CHARS_HARD_LIMIT, max_chars)
        if symbol is not None:
            self._ensure_symbol_indexed(symbol)
            qualname = self._resolve_symbol_qualified_name(symbol)
            if qualname is None:
                return None
            defn = self.symbol_index.get(qualname)
        elif file_path is not None:
            self._ensure_file_indexed(file_path)
            path = Path(file_path).resolve()
            module_node = self.outline_index.get(path)
            defn = module_node.symbol if module_node is not None else None
        else:
            return None

        if defn is None:
            return None

        doc = defn.docstring
        if doc and len(doc) > max_chars:
            doc = doc[:max_chars] + "…"

        init_doc = defn.init_docstring
        if init_doc and len(init_doc) > max_chars:
            init_doc = init_doc[:max_chars] + "…"

        return {
            "name": defn.name,
            # "qualified_name": defn.qualified_name,
            "kind": defn.kind,
            "type": defn.type_annotation,
            "file_path": str(defn.file_path),
            # "line": defn.line,
            # "column": defn.column,
            "docstring": doc,
            "init_docstring": init_doc,
        }

    def get_definition_full(
        self,
        symbol: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        max_chars: int = 5000,
    ) -> Optional[Dict[str, Any]]:
        """Return a full definition (including source) for a symbol or module.

        Parameters
        ----------
        symbol : str, optional
            Symbol name or qualified name for which to retrieve the
            definition. If provided, this takes precedence over `file_path`.
        file_path : str or pathlib.Path, optional
            Path to a Python file. If provided and `symbol` is ``None``,
            the module-level definition is returned.
        max_chars : int, optional
            Maximum length for the returned source text. Longer text is
            truncated with a trailing marker. Default is 5000.

        Returns
        -------
        dict or None
            Dictionary containing full definition information for the symbol,
            or ``None`` if the symbol cannot be found.
        """
        max_chars = min(MAX_CHARS_HARD_LIMIT, max_chars)
        if symbol is not None:
            self._ensure_symbol_indexed(symbol)
            qualname = self._resolve_symbol_qualified_name(symbol)
            if qualname is None:
                return None
            defn = self.symbol_index.get(qualname)
        elif file_path is not None:
            self._ensure_file_indexed(file_path)
            path = Path(file_path).resolve()
            module_node = self.outline_index.get(path)
            defn = module_node.symbol if module_node is not None else None
        else:
            return None

        if defn is None:
            return None

        source = defn.full_source or ""
        if len(source) > max_chars:
            source = source[:max_chars] + "\n…(truncated)"

        return {
            "name": defn.name,
            # "qualified_name": defn.qualified_name,
            "kind": defn.kind,
            "type": defn.type_annotation,
            "file_path": str(defn.file_path),
            "line": defn.line,
            "column": defn.column,
            "docstring": defn.docstring,
            "init_docstring": defn.init_docstring,
            "source": source,
        }

    def _serialize_outline_node(
        self,
        node: OutlineNode,
        max_chars: int,
    ) -> Dict[str, Any]:
        """Convert an outline node into a serializable dictionary.

        Parameters
        ----------
        node : OutlineNode
            Node to serialize.
        max_chars : int
            Maximum length for docstring fields.

        Returns
        -------
        dict
            Serialized representation of the node and its children.
        """
        max_chars = min(MAX_CHARS_HARD_LIMIT, max_chars)
        defn = node.symbol
        doc = defn.docstring
        if doc and len(doc) > max_chars:
            doc = doc[:max_chars] + "…"

        init_doc = defn.init_docstring
        if init_doc and len(init_doc) > max_chars:
            init_doc = init_doc[:max_chars] + "…"

        return {
            "name": defn.name,
            # "qualified_name": defn.qualified_name,
            "kind": defn.kind,
            "type": defn.type_annotation,
            "file_path": str(defn.file_path),
            "line": defn.line,
            "column": defn.column,
            "docstring": doc,
            "init_docstring": init_doc,
            "children": [
                self._serialize_outline_node(child, max_chars)
                for child in node.children
            ],
        }

    def get_outline(
        self,
        symbol: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        max_chars: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Return an outline tree for a symbol or module.

        Parameters
        ----------
        symbol : str, optional
            Symbol name or qualified name. If provided, the outline is
            generated from that symbol node downward.
        file_path : str or pathlib.Path, optional
            File path. If provided and `symbol` is ``None``, the outline
            for the module corresponding to that file is returned.
        max_chars : int, optional
            Maximum length for docstring fields. Default is 1000.

        Returns
        -------
        list of dict
            List of serialized child nodes under the requested symbol or
            module. If neither symbol nor file path can be resolved, an
            empty list is returned.
        """
        max_chars = min(MAX_CHARS_HARD_LIMIT, max_chars)
        node: Optional[OutlineNode]

        if symbol is not None:
            self._ensure_symbol_indexed(symbol)
            qualname = self._resolve_symbol_qualified_name(symbol)
            if qualname is None:
                return []
            node = self._node_index.get(qualname)
            if node is None:
                return []
        elif file_path is not None:
            self._ensure_file_indexed(file_path)
            path = Path(file_path).resolve()
            node = self.outline_index.get(path)
            if node is None:
                return []
        else:
            return []

        # We return children to avoid wrapping the root/module itself,
        # but you can change this if you prefer.
        return [
            self._serialize_outline_node(child, max_chars) for child in node.children
        ]

    def get_references(
        self,
        symbol: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        max_chars: int = 300,
    ) -> List[Dict[str, Any]]:
        """Return symbol references filtered by symbol or file path.

        Parameters
        ----------
        symbol : str, optional
            Symbol name for which to retrieve references. If provided, this
            takes precedence over `file_path`.
        file_path : str or pathlib.Path, optional
            File path for which to list all references occurring in that
            file, regardless of symbol. Ignored if `symbol` is provided.
        max_chars : int, optional
            Maximum length for context strings. Default is 300.

        Returns
        -------
        list of dict
            List of reference dictionaries containing file path, location
            and optional context. If neither `symbol` nor `file_path` can
            be resolved, an empty list is returned.

        Notes
        -----
        In sparse mode, only files that have been indexed so far contribute
        references. Additional references may appear as more files are
        indexed over time.
        """
        max_chars = min(MAX_CHARS_HARD_LIMIT, max_chars)
        results: List[SymbolReference]

        if symbol is not None:
            self._ensure_symbol_indexed(symbol)
            results = self._refs_by_symbol.get(symbol, [])
        elif file_path is not None:
            self._ensure_file_indexed(file_path)
            path = Path(file_path).resolve()
            results = [ref for ref in self.references if ref.file_path == path]
        else:
            return []

        out: List[Dict[str, Any]] = []
        for ref in results:
            ctx = ref.context
            if ctx and len(ctx) > max_chars:
                ctx = ctx[:max_chars] + "…"
            out.append(
                {
                    "symbol": ref.symbol,
                    "file_path": str(ref.file_path),
                    "line": ref.line,
                    "column": ref.column,
                    # "context": ctx,
                }
            )
        return out

    def filter_symbols(
        self,
        name: Optional[str] = None,
        kind: Optional[str] = None,
        type_hint: Optional[str] = None,
        file_path: Optional[Union[str, Path]] = None,
        max_results: int = 50,
        max_chars: int = 500,
    ) -> List[Dict[str, Any]]:
        """Filter and search symbols in the project index by name, kind, type hint, or file.

        Parameters
        ----------
        name : str, optional
            Substring (case-insensitive) to match in symbol names.
        kind : str, optional
            Filter by symbol kind (e.g., 'class', 'function', 'variable').
        type_hint : str, optional
            Substring to match within symbol type annotations.
        file_path : str or pathlib.Path, optional
            Restrict search to symbols defined in this file.
        max_results : int, optional
            Maximum number of results to return. Default is 50.
        max_chars : int, optional
            Maximum length for text fields (e.g. docstrings). Default is 500.

        Returns
        -------
        list of dict
            A list of filtered symbol information dictionaries.
        """
        max_chars = min(MAX_CHARS_HARD_LIMIT, max_chars)
        matches: List[Dict[str, Any]] = []

        # Ensure file is indexed if filtering by file
        if file_path is not None:
            self._ensure_file_indexed(file_path)
            path = Path(file_path).resolve()
            candidates = [d for d in self.symbol_index.values() if d.file_path == path]
        else:
            candidates = list(self.symbol_index.values())

        # Apply filters
        for defn in candidates:
            if name and name.lower() not in defn.name.lower():
                continue
            if kind and defn.kind.lower() != kind.lower():
                continue
            if type_hint:
                ann = defn.type_annotation or ""
                if type_hint.lower() not in ann.lower():
                    continue

            doc = defn.docstring or ""
            if len(doc) > max_chars:
                doc = doc[:max_chars] + "…"

            init_doc = defn.init_docstring or ""
            if len(init_doc) > max_chars:
                init_doc = init_doc[:max_chars] + "…"

            matches.append(
                {
                    "name": defn.name,
                    # "qualified_name": defn.qualified_name,
                    "kind": defn.kind,
                    "type": defn.type_annotation,
                    "file_path": str(defn.file_path),
                    # "line": defn.line,
                    # "column": defn.column,
                    # "docstring": doc,
                    # "init_docstring": init_doc,
                }
            )

            if len(matches) >= max_results:
                break

        return matches
