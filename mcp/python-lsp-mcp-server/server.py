# server.py
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine import LSPEngine
from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger as get_fastmcp_logger
from loguru import logger


# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
def _configure_loguru(log_level: str = "INFO") -> None:
    """Configure global loguru logging.

    Parameters
    ----------
    log_level : str, optional
        Minimum log level for emitted logs. Typical values are ``"DEBUG"``,
        ``"INFO"``, ``"WARNING"``, and ``"ERROR"``. Default is ``"INFO"``.
    """
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "| <level>{level: <8}</level> "
            "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
    )


class _InterceptHandler(logging.Handler):
    """Bridge standard logging records into loguru.

    This handler can be attached to the root logging logger so that log
    messages emitted by FastMCP (which uses `logging`) are forwarded to
    loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = "INFO"
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def _configure_standard_logging() -> None:
    """Configure standard logging to route through loguru.

    This allows FastMCP's internal logging (which uses the standard
    `logging` module) to appear in the same log stream as user logs
    emitted via loguru.
    """
    logging.root.handlers = []
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(_InterceptHandler())


# ---------------------------------------------------------------------
# MCP server and engine wiring
# ---------------------------------------------------------------------
mcp = FastMCP("python-lsp-mcp-server", json_response=True)

# Global engine instance, initialised in `main`.
_ENGINE: Optional[LSPEngine] = None


def _get_engine() -> LSPEngine:
    """Return the global LSPEngine instance.

    Returns
    -------
    LSPEngine
        Currently configured engine instance.

    Raises
    ------
    RuntimeError
        If the engine has not been initialised. This should not occur if
        `main()` has been executed prior to starting the MCP server.
    """
    if _ENGINE is None:
        raise RuntimeError("LSPEngine not initialised. Did you run main()?")
    return _ENGINE


# ---------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------
@mcp.tool
async def definition_short(
    symbol: Optional[str] = None,
    file_path: Optional[str] = None,
    max_chars: int = 1000,
) -> Optional[Dict[str, Any]]:
    """Retrieve concise info about a symbol limited to name, docstring and location.

    Parameters
    ----------
    symbol : str, optional
        Symbol name or qualified name. Takes precedence over `file_path`.
    file_path : str, optional
        Python file path if `symbol` is not given.
    max_chars : int, optional
        Maximum docstring length (default: 1000).
    """
    engine = _get_engine()
    logger.info(
        "definition_short called with symbol={!r}, file_path={!r}", symbol, file_path
    )
    return engine.get_definition_short(
        symbol=symbol, file_path=file_path, max_chars=max_chars
    )


@mcp.tool
async def definition_full(
    symbol: Optional[str] = None,
    file_path: Optional[str] = None,
    max_chars: int = 5000,
) -> Optional[Dict[str, Any]]:
    """Retrieve full source and metadata for a symbol or module.

    Parameters
    ----------
    symbol : str, optional
        Symbol name or qualified name. Takes precedence over `file_path`.
    file_path : str, optional
        Python file path if `symbol` is not given.
    max_chars : int, optional
        Maximum source length (default: 5000).
    """
    engine = _get_engine()
    logger.info(
        "definition_full called with symbol={!r}, file_path={!r}", symbol, file_path
    )
    return engine.get_definition_full(
        symbol=symbol, file_path=file_path, max_chars=max_chars
    )


@mcp.tool
async def outline(
    symbol: Optional[str] = None,
    file_path: Optional[str] = None,
    max_chars: int = 1000,
) -> List[Dict[str, Any]]:
    """List symbols and structure under a symbol or module. WARNING: Large output content.

    Parameters
    ----------
    symbol : str, optional
        Symbol name or qualified name.
    file_path : str, optional
        Python file path if `symbol` is not given.
    max_chars : int, optional
        Maximum docstring length (default: 1000).
    """
    engine = _get_engine()
    logger.info("outline called with symbol={!r}, file_path={!r}", symbol, file_path)
    return engine.get_outline(symbol=symbol, file_path=file_path, max_chars=max_chars)


@mcp.tool
async def references(
    symbol: Optional[str] = None,
    file_path: Optional[str] = None,
    max_chars: int = 300,
) -> List[Dict[str, Any]]:
    """List references for a symbol or file.

    Parameters
    ----------
    symbol : str, optional
        Symbol name for which to retrieve references.
    file_path : str, optional
        Python file path if `symbol` is not given.
    max_chars : int, optional
        Maximum context length (default: 300).
    """
    engine = _get_engine()
    logger.info("references called with symbol={!r}, file_path={!r}", symbol, file_path)
    return engine.get_references(
        symbol=symbol, file_path=file_path, max_chars=max_chars
    )


@mcp.tool
async def filter_symbols(
    name: Optional[str] = None,
    kind: Optional[str] = None,
    type_hint: Optional[str] = None,
    max_results: int = 50,
    max_chars: int = 500,
) -> List[dict]:
    """Filter symbols by name, kind, or type hint.

    Parameters
    ----------
    name : str, optional
        Substring or regex in symbol name.
    kind : str, optional
        Symbol kind ('class', 'function', etc.).
    type_hint : str, optional
        Substring in type annotation.
    max_results : int, optional
        Maximum number of results (default: 50).
    max_chars : int, optional
        Maximum docstring length (default: 500).
    """
    engine = _get_engine()
    logger.info(
        "filter_symbols called with name={!r}, kind={!r}, type_hint={!r}",
        name,
        kind,
        type_hint,
    )
    return engine.filter_symbols(
        name=name,
        kind=kind,
        type_hint=type_hint,
        max_results=max_results,
        max_chars=max_chars,
    )


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    """Entry point to start the MCP server.

    The project root and server transport can be configured via command
    line arguments.

    Examples
    --------
    Run with stdio (typical for MCP integration):

    .. code-block:: bash

        python -m src.server --project-root . --transport stdio

    Run with HTTP transport:

    .. code-block:: bash

        python -m src.server --project-root . --transport streamable-http --port 8000
    """
    parser = argparse.ArgumentParser(description="Python LSP-style MCP server.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=os.getcwd(),
        help="Root directory of the Python project to index.",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "streamable-http"],
        help="Transport mechanism for the MCP server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for HTTP transport.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level for loguru (e.g. DEBUG, INFO).",
    )
    args = parser.parse_args()

    _configure_loguru(log_level=args.log_level)
    _configure_standard_logging()

    # Optional: get a FastMCP logger (routes through std logging, which we already
    # bridged to loguru). You can use it if you want namespaced logs.
    fmcp_logger = get_fastmcp_logger("server")
    fmcp_logger.info("Starting Python LSP MCP server")

    global _ENGINE
    _ENGINE = LSPEngine(project_root=Path(args.project_root))

    if args.transport == "streamable-http":
        logger.info(
            "Running MCP server with HTTP transport on {}:{}",
            args.host,
            args.port,
        )
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        logger.info("Running MCP server with stdio transport")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
