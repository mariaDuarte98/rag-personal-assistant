from typing import Callable, Dict, Any, List
import os
import subprocess
import sys


def open_file(path: str) -> str:
    """Open a file using the OS default application."""
    if not os.path.exists(path):
        return "File not found."

    subprocess.run(["open", path])  # Windows
    return f"Opened file: {path}"


def list_files(directory: str) -> str:
    """List files in a directory."""
    if not os.path.exists(directory):
        return []

    return f"The files in {directory} are: {os.listdir(directory)}"


def find_and_open(name: str, search_roots: list[str]) -> str:
    """
    Search for a file or folder by name and open it.
    """
    name = name.lower()

    for root in search_roots:
        if not os.path.exists(root):
            continue

        for current_root, dirs, files in os.walk(root):
            # Check folders
            for d in dirs:
                if d.lower() == name:
                    path = os.path.join(current_root, d)
                    return open_file(path)

            # Check files
            for f in files:
                if f.lower() == name:
                    path = os.path.join(current_root, f)
                    return open_file(path)

    return "No matching file or folder found."


def resolve_allowed_domains(domains: List[str]) -> List[str]:
    """
    Convert logical domain names (desktop, documents, downloads)
    into real filesystem paths.
    """
    home = os.path.expanduser("~")
    domain_list = ["Desktop", "Documents", "Downloads"] 
    domain_list += domains

    resolved_paths = []

    for domain in domain_list:
        path = os.path.join(home, domain)
        if path and os.path.exists(path):
            resolved_paths.append(path)

    return resolved_paths


def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    if not os.path.exists(path):
        return "File not found."

    if not os.path.isfile(path):
        return "Path is not a file."

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file: {e}"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file with the given content."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"File written: {path}"
    except Exception as e:
        return f"Failed to write file: {e}"


def append_file(path: str, content: str) -> str:
    """Append content to a file (creates it if missing)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            f.write(content)

        return f"Content appended to: {path}"
    except Exception as e:
        return f"Failed to append file: {e}"


def create_directory(path: str) -> str:
    """Create a directory if it does not exist."""
    try:
        os.makedirs(path, exist_ok=True)
        return f"Directory ensured: {path}"
    except Exception as e:
        return f"Failed to create directory: {e}"

#def play_spotify() -> str:
#    """Open Spotify application."""
#    subprocess.Popen(["spotify"])
#    return "Spotify opened."

ACTION_REGISTRY: Dict[str, Callable[..., Any]] = {
    "open_file": open_file,
    "list_files": list_files,
    "resolve_allowed_domains": resolve_allowed_domains,
    "find_and_open": find_and_open,
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "create_directory": create_directory,
    #"play_spotify": play_spotify,
}
