import os
import sys
from pathlib import Path


SOFA_PYTHON_VERSION = (3, 10)
_WINDOWS_DLL_HANDLES = []
_REGISTERED_DLL_DIRS: set[str] = set()


def validate_sofa_python() -> None:
    if sys.version_info[:2] == SOFA_PYTHON_VERSION:
        return

    expected = ".".join(str(part) for part in SOFA_PYTHON_VERSION)
    current = sys.version.split()[0]
    raise RuntimeError(
        "SOFA bindings in this lab require Python "
        f"{expected}, but the current interpreter is Python {current}. "
        "Use a Python 3.10 environment for SOFA-based scripts."
    )


def _venv_site_packages_candidates(project_dir: Path) -> list[Path]:
    python_tag = f"python{SOFA_PYTHON_VERSION[0]}.{SOFA_PYTHON_VERSION[1]}"
    return [
        project_dir / ".venv" / "Lib" / "site-packages",
        project_dir / ".venv" / "lib" / "site-packages",
        project_dir / ".venv" / "lib" / python_tag / "site-packages",
        project_dir / ".venv" / "lib64" / python_tag / "site-packages",
    ]


def _resolve_sofa_root(project_dir: Path) -> Path:
    configured_root = os.environ.get("SOFA_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()

    candidate_roots: list[Path] = []
    workspace_root = project_dir.parents[2] if len(project_dir.parents) >= 3 else None
    if workspace_root is not None:
        candidate_roots.append(workspace_root / "resources" / "sofa")

    if os.name == "nt":
        appdata = os.getenv("LOCALAPPDATA")
        if appdata:
            candidate_roots.append(Path(appdata) / "Programs" / "emio-labs" / "resources" / "sofa")
    else:
        candidate_roots.append(Path("/opt/emio-labs/resources/sofa"))

    if not candidate_roots:
        candidate_roots.append(project_dir / "resources" / "sofa")

    sofa_root = next((path for path in candidate_roots if path.exists()), candidate_roots[0])
    os.environ["SOFA_ROOT"] = str(sofa_root)
    return sofa_root


def _sofa_python_path_candidates(sofa_root: Path) -> list[Path]:
    candidates = [
        sofa_root / "plugins" / "SofaPython3" / "lib" / "python3" / "site-packages",
        sofa_root / "plugins" / "SofaPython3" / "lib" / "site-packages",
        sofa_root / "plugins" / "SofaPython3" / "Lib" / "site-packages",
        sofa_root / "bin" / "python" / "Lib" / "site-packages",
        sofa_root / "bin" / "python" / "lib" / "site-packages",
    ]

    sofa_python_plugin = sofa_root / "plugins" / "SofaPython3"
    if sofa_python_plugin.is_dir():
        candidates.extend(sorted(sofa_python_plugin.glob("**/site-packages")))

    unique_candidates = []
    seen = set()
    for path in candidates:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        unique_candidates.append(path)
    return unique_candidates


def _register_windows_dll_directories(sofa_root: Path) -> None:
    if os.name != "nt":
        return

    dll_candidates = [
        sofa_root / "bin",
        sofa_root / "lib",
        sofa_root / "plugins" / "SofaPython3" / "bin",
        sofa_root / "plugins" / "SofaPython3" / "lib",
    ]

    plugins_root = sofa_root / "plugins"
    if plugins_root.is_dir():
        dll_candidates.extend(sorted(plugins_root.glob("*/bin")))

    existing_dirs = [path for path in dll_candidates if path.exists()]
    if not existing_dirs:
        return

    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep) if current_path else []
    existing_path_entries = {entry.lower() for entry in path_entries if entry}
    prepended_entries = []

    for dll_dir in existing_dirs:
        dll_dir_str = str(dll_dir)
        dll_dir_key = dll_dir_str.lower()
        if dll_dir_key not in existing_path_entries:
            prepended_entries.append(dll_dir_str)
            existing_path_entries.add(dll_dir_key)

        if hasattr(os, "add_dll_directory") and dll_dir_key not in _REGISTERED_DLL_DIRS:
            try:
                _WINDOWS_DLL_HANDLES.append(os.add_dll_directory(dll_dir_str))
                _REGISTERED_DLL_DIRS.add(dll_dir_key)
            except (FileNotFoundError, OSError):
                continue

    if prepended_entries:
        os.environ["PATH"] = os.pathsep.join([*prepended_entries, *path_entries])


def bootstrap_sofa_python() -> Path:
    project_dir = Path(__file__).resolve().parent.parent
    assets_dir = project_dir.parent.parent
    sofa_root = _resolve_sofa_root(project_dir)
    _register_windows_dll_directories(sofa_root)

    for path in (
        project_dir,
        assets_dir,
        *_sofa_python_path_candidates(sofa_root),
        *_venv_site_packages_candidates(project_dir),
    ):
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return sofa_root


def bootstrap_and_validate_sofa() -> Path:
    validate_sofa_python()
    return bootstrap_sofa_python()
