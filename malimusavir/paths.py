"""Where files live, whether running from source or from a frozen .exe.

PyInstaller unpacks bundled files into a temporary directory and deletes it on exit.
That is fine for code, the schema, the web assets and the trained model -- all read-only
things that ship with the build. It is catastrophic for the database: written there, a
user's invoices would be silently discarded every time they closed the program.

So the two are kept apart deliberately:

    resource()   read-only, ships with the build, lives in _MEIPASS when frozen
    user_data()  writable, lives beside the .exe so it survives and can be backed up

Running from source both resolve to the repository root, which is what they were before
and why nothing about development changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The repository root when running from source. paths.py sits in malimusavir/.
_SOURCE_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def resource_root() -> Path:
    """Where bundled read-only files were unpacked."""
    if is_frozen():
        return Path(sys._MEIPASS)      # noqa: SLF001 - PyInstaller's documented API
    return _SOURCE_ROOT


def user_data_root() -> Path:
    """Where files the user owns belong: next to the executable they launched.

    sys.executable is the .exe itself when frozen, so the database sits beside it in a
    folder the user chose -- visible, backup-able, and still there tomorrow.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _SOURCE_ROOT


def resource(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def user_data(*parts: str) -> Path:
    return user_data_root().joinpath(*parts)
