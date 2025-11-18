import subprocess
from pathlib import Path

def run_jwst_merge(recipe_path: Path, repo_path: Path):
    """
    Example wrapper that calls a JWST-Merge script.
    Adjust script name to match your repo's entrypoint.
    """
    recipe_path = Path(recipe_path).resolve()
    repo_path = Path(repo_path).resolve()
    # Example call -- adjust to the actual JWST-Merge script in your repo
    script = repo_path / "Step 4 Basic Processing Script.py"
    if not script.exists():
        raise FileNotFoundError(f"Expected JWST-Merge script at {script}")
    cmd = ["python", str(script), str(recipe_path)]
    subprocess.check_call(cmd, cwd=str(repo_path))
