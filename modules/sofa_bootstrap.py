import os
import sys
from pathlib import Path


SOFA_PYTHON_VERSION = (3, 10)


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


def bootstrap_sofa_python() -> Path:
    project_dir = Path(__file__).resolve().parent.parent
    assets_dir = project_dir.parent.parent

    if os.name == "posix":
        sofa_root = Path(os.environ.setdefault("SOFA_ROOT", "/opt/emio-labs/resources/sofa"))
    else:
        appdata = os.getenv("LOCALAPPDATA", "")
        sofa_root = Path(
            os.environ.setdefault(
                "SOFA_ROOT",
                os.path.join(appdata, "Programs", "emio-labs", "resources", "sofa"),
            )
        )

    sofa_python = (
        sofa_root / "plugins" / "SofaPython3" / "lib" / "python3" / "site-packages"
    )

    for path in (project_dir, assets_dir, sofa_python):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return sofa_root


def bootstrap_and_validate_sofa() -> Path:
    validate_sofa_python()
    return bootstrap_sofa_python()
