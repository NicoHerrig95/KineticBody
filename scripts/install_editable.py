# scripts/install_editable.py
import subprocess
import os
import sys


""" 
Installs project in editable mode.
Mainly for project maintainability (import managing and base_dir setting)
"""

def install_project_editable():
    # Get the path to the project root (one level up from scripts/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print(f"Installing project in editable mode from: {project_root}")

    # Build the pip install command
    command = [sys.executable,  "-m", "pip", "install", "-e", project_root]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Project installed in editable mode successfully.")
    else:
        print("❌ Installation failed!")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)

if __name__ == "__main__":
    install_project_editable()