import subprocess
from pathlib import Path

# Copyright (C) <2026>, Streamline Aerospace. All rights reserved.

def run_nastran(
    bdf_path: str | Path,
    nastran_path: str | Path,
    work_dir: str | Path | None = None,
    nastran_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Runs nastran.exe directly on an existing BDF file.

    Parameters
    ----------
    bdf_path : str or Path
        Path to the existing .bdf file (e.g. 'xRUN_RV10_WING_flutter_15kft.bdf').
    nastran_path : str or Path, optional
        Path to the nastran.exe executable.
    work_dir : str or Path, optional
        Working directory to execute NASTRAN in. If None, uses the parent directory
        of the BDF file so relative INCLUDE cards resolve properly.
    nastran_args : list[str], optional
        List of auxiliary command-line keywords passed directly to the solver
        (e.g., ["mem=8GB", "sdir=C:\\scratch", "scr=yes", "parallel=4"]).

    Returns
    -------
    subprocess.CompletedProcess
        Result of the execution containing returncode, stdout, and stderr.
    """
    bdf_file = Path(bdf_path).resolve()
    if not bdf_file.is_file():
        raise FileNotFoundError(f"BDF input file not found: {bdf_file}")

    # Default to the directory where the BDF resides so INCLUDE cards find their files
    execution_dir = Path(work_dir).resolve() if work_dir else bdf_file.parent

    # Command: nastran.exe <filename.bdf>
    cmd = [str(nastran_path), bdf_file.name]

    if nastran_args:
        # Extend the command with auxiliary arguments (stripping leading/trailing spaces)
        cmd.extend(arg.strip() for arg in nastran_args if arg.strip())

    print(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=execution_dir,
        capture_output=True,
        text=True,
        check=True,  # Raises CalledProcessError if returncode != 0
    )

    return result