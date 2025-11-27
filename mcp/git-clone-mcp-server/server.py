# server.py
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger as get_fastmcp_logger
from loguru import logger


# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
def _configure_loguru(log_level: str = "INFO") -> None:
    """configure global loguru logging.

    Parameters
    ----------
    log_level : str, optional
        minimum log level for emitted logs. typical values are "DEBUG",
        "INFO", "WARNING", and "ERROR". default is "INFO".
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
    """bridge standard logging records into loguru.

    this handler can be attached to the root logging logger so that log
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
    """configure standard logging to route through loguru.

    this allows FastMCP's internal logging (which uses the standard
    `logging` module) to appear in the same log stream as user logs
    emitted via loguru.
    """
    logging.root.handlers = []
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(_InterceptHandler())


# ---------------------------------------------------------------------
# MCP server and engine wiring
# ---------------------------------------------------------------------
mcp = FastMCP("git-clone-mcp-server", json_response=True)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _validate_github_repo(owner: str, repo: str) -> Dict[str, Any]:
    """validate if a github repository exists.

    Parameters
    ----------
    owner : str
        repository owner username or organization name
    repo : str
        repository name

    Returns
    -------
    Dict[str, Any]
        dictionary with 'valid' (bool) and 'message' (str) keys
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    try:
        response = requests.get(api_url, timeout=10)

        if response.status_code == 200:
            return {
                "valid": True,
                "message": f"repository {owner}/{repo} exists",
                "data": response.json(),
            }
        elif response.status_code == 404:
            return {"valid": False, "message": f"repository {owner}/{repo} not found"}
        else:
            return {
                "valid": False,
                "message": f"unexpected status code {response.status_code} for {owner}/{repo}",
            }
    except requests.exceptions.RequestException as e:
        return {"valid": False, "message": f"failed to validate repository: {str(e)}"}


def _validate_target_directory(target_path: str) -> Dict[str, Any]:
    """validate if the target directory path is valid.

    Parameters
    ----------
    target_path : str
        path to the target directory where repo will be cloned

    Returns
    -------
    Dict[str, Any]
        dictionary with 'valid' (bool) and 'message' (str) keys
    """
    path_obj = Path(target_path)

    # check if parent directory exists
    parent_dir = path_obj.parent
    if not parent_dir.exists():
        return {
            "valid": False,
            "message": f"parent directory {parent_dir} does not exist",
        }

    # check if parent is a directory
    if not parent_dir.is_dir():
        return {
            "valid": False,
            "message": f"parent path {parent_dir} is not a directory",
        }

    # check if target path already exists
    if path_obj.exists():
        return {"valid": False, "message": f"target path {target_path} already exists"}

    # check if parent directory is writable
    if not os.access(parent_dir, os.W_OK):
        return {
            "valid": False,
            "message": f"no write permission for parent directory {parent_dir}",
        }

    return {"valid": True, "message": f"target path {target_path} is valid"}


# ---------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------
@mcp.tool
async def clone_github_repo(
    repo_name: str, target_path: str, branch: str = "main"
) -> Dict[str, Any]:
    """clone a public github repository to a specified local directory.

    Parameters
    ----------
    repo_name : str
        repository name in the format "owner/repo" (e.g., "torvalds/linux")
    target_path : str
        absolute or relative path where the repository should be cloned
    branch : str, optional
        branch name to clone, defaults to "main"
    """
    logger.info(f"attempting to clone {repo_name} (branch: {branch}) to {target_path}")

    # parse repo_name into owner and repo
    parts = repo_name.split("/")
    if len(parts) != 2:
        error_msg = (
            f"invalid repo_name format: {repo_name}. expected format: owner/repo"
        )
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "repo_path": None,
            "branch": branch,
        }

    owner, repo = parts

    # validate github repository
    logger.debug(f"validating github repository {owner}/{repo}")
    repo_validation = _validate_github_repo(owner, repo)
    if not repo_validation["valid"]:
        logger.error(f"repository validation failed: {repo_validation['message']}")
        return {
            "success": False,
            "message": repo_validation["message"],
            "repo_path": None,
            "branch": branch,
        }

    logger.info(f"repository {owner}/{repo} validated successfully")

    # validate target directory
    logger.debug(f"validating target directory: {target_path}")
    dir_validation = _validate_target_directory(target_path)
    if not dir_validation["valid"]:
        logger.error(f"directory validation failed: {dir_validation['message']}")
        return {
            "success": False,
            "message": dir_validation["message"],
            "repo_path": None,
            "branch": branch,
        }

    logger.info(f"target directory {target_path} validated successfully")

    # construct github URL
    repo_url = f"https://github.com/{owner}/{repo}.git"

    # execute git clone command
    try:
        logger.info(f"executing git clone from {repo_url}")
        result = subprocess.run(
            ["git", "clone", "--branch", branch, repo_url, target_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        logger.info(f"successfully cloned {repo_name} to {target_path}")
        return {
            "success": True,
            "message": f"successfully cloned {repo_name} (branch: {branch}) to {target_path}",
            "repo_path": str(Path(target_path).resolve()),
            "branch": branch,
            "git_output": result.stdout,
        }

    except subprocess.CalledProcessError as e:
        error_msg = f"git clone failed: {e.stderr}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "repo_path": None,
            "branch": branch,
            "git_error": e.stderr,
        }
    except subprocess.TimeoutExpired:
        error_msg = "git clone timed out."
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "repo_path": None,
            "branch": branch,
        }
    except FileNotFoundError:
        error_msg = "git command not found. please ensure git is installed and in PATH"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "repo_path": None,
            "branch": branch,
        }
    except Exception as e:
        error_msg = f"unexpected error during clone: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "repo_path": None,
            "branch": branch,
        }


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main() -> None:
    """entry point to start the MCP server.

    the server transport can be configured via command line arguments.

    Examples
    --------
    run with stdio (typical for MCP integration):

    .. code-block:: bash

        python -m server --transport stdio

    run with HTTP transport:

    .. code-block:: bash

        python -m server --transport streamable-http --port 8000
    """
    parser = argparse.ArgumentParser(description="git clone MCP server.")
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "streamable-http"],
        help="transport mechanism for the MCP server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host for HTTP transport.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port for HTTP transport.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="log level for loguru (e.g. DEBUG, INFO).",
    )
    args = parser.parse_args()

    _configure_loguru(log_level=args.log_level)
    _configure_standard_logging()

    fmcp_logger = get_fastmcp_logger("server")
    fmcp_logger.info("starting git clone MCP server")

    if args.transport == "streamable-http":
        logger.info(
            "running MCP server with HTTP transport on {}:{}",
            args.host,
            args.port,
        )
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        logger.info("running MCP server with stdio transport")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
