# PyInstaller build for Mali Müşavir.
#
#     .venv\Scripts\pyinstaller.exe MaliMusavir.spec
#
# Produces dist/MaliMusavir/MaliMusavir.exe -- a folder build, not a single file, and
# deliberately so. A one-file exe unpacks ~400 MB of pandas, numpy and scikit-learn to a
# temp directory on *every* launch, which costs 10-20 seconds each time and trips
# antivirus heuristics. The folder starts in about a second and can still be zipped or
# shortcut-ed.
#
# The database is NOT bundled: malimusavir/paths.py puts it next to the .exe, so a
# user's invoices survive closing the program. Bundling it would write to PyInstaller's
# temp extraction directory, which is deleted on exit.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Read-only files the app loads at runtime by path. These land beside the frozen
# modules, which is where paths.resource() looks.
datas = [
    ("schema.sql", "."),
    ("web", "web"),
    ("malimusavir/models", "malimusavir/models"),
]

# pdfplumber and pdfminer ship character-mapping tables as package data; without them
# text extraction fails at import time rather than politely.
datas += collect_data_files("pdfminer")
datas += collect_data_files("pdfplumber")

hiddenimports = [
    # uvicorn resolves its loop and protocol implementations by string name, so the
    # analyser cannot see them.
    "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on",
]
# scikit-learn loads its estimators dynamically when unpickling the trained head.
hiddenimports += collect_submodules("sklearn.utils")
hiddenimports += collect_submodules("sklearn.linear_model")

analysis = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # Nothing here is used, and each drags in tens of megabytes.
    excludes=["matplotlib", "tkinter", "PyQt5", "PySide2", "IPython", "notebook",
              "pytest", "reportlab"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="MaliMusavir",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # console=True on purpose. The app *is* a local server: the window shows which port
    # it is on, whether Foundry started, and any traceback. A windowed build would fail
    # silently, and "it does not open" is the least debuggable bug report there is.
    console=True,
    disable_windowed_traceback=False,
)

COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="MaliMusavir",
)
