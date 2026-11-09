import tkinter as tk 
from tkinter import filedialog
from pathlib import Path

def select_directory(
    prompt_title: str = "Select a Directory", initial_dir: str = "/"
) -> str:
    """Opens a file dialog window prompting the user to select a directory.

    Parameters
    ----------
    prompt_title : str, optional
        The title displayed on the dialog window (default: "Select a Directory").
    initial_dir : str, optional
        The directory path where the dialog opens initially (default: "/").

    Returns
    -------
    str
        The absolute path to the selected directory, or an empty string if cancelled.
    """
    # Create the root window and hide it so only the dialog appears
    root = tk.Tk()
    root.withdraw()

    # Bring dialog to the front (useful across different operating systems)
    root.attributes("-topmost", True)

    # Prompt user to choose a directory
    selected_path = filedialog.askdirectory(
        title=prompt_title, initialdir=initial_dir
    )

    # Destroy the hidden root window to clean up resources
    root.destroy()

    return selected_path


def find_bdf_files(
    directory: str | Path, recursive: bool = False
) -> list[Path]:
    """Searches a directory and returns a list of paths to all .bdf files.

    Parameters
    ----------
    directory : str or Path
        The directory path to search.
    recursive : bool, optional
        If True, searches all subdirectories recursively. Defaults to False.

    Returns
    -------
    list[Path]
        A sorted list of Path objects for all matching .bdf files (case-insensitive).
    """
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    # Use rglob for recursive search, glob for top-level search
    pattern = "**/*" if recursive else "*"

    # Case-insensitive check (.bdf, .BDF, etc.)
    bdf_files = [
        f for f in dir_path.glob(pattern) if f.is_file() and f.suffix.lower() == ".bdf"
    ]

    return sorted(bdf_files)


def get_subdirectories(
    directory: str | Path, full_path: bool = True
) -> list[str]:
    """Scans a directory and returns a list of folders within that directory.

    Parameters
    ----------
    directory : str or Path
        The directory path to scan.
    full_path : bool, optional
        If True, returns absolute/resolved paths. If False, returns folder names.
        Defaults to True.

    Returns
    -------
    list[str]
        A sorted list of directory paths (or folder names). Returns an empty list
        if no folders are present.
    """
    dir_path = Path(directory).resolve()

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    # Filter only directory entries (ignoring files)
    subfolders = [item for item in dir_path.iterdir() if item.is_dir()]

    if full_path:
        return sorted(str(p) for p in subfolders)
    return sorted(p.name for p in subfolders)