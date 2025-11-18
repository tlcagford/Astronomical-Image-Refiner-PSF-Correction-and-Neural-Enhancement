import subprocess
from pathlib import Path

def run_wfc3_psf(recipe_path: Path, repo_path: Path):
    """
    Example wrapper that calls the wfc3-psf pipeline.
    Replace 'run_wfc3_psf.py' with the real script name in your repo.
    """
    recipe_path = Path(recipe_path).resolve()
    repo_path = Path(repo_path).resolve()
    script = repo_path / "run_wfc3_psf.py"
    if not script.exists():
        # If the real repo uses other entrypoint names, adapt here or call a function.
        raise FileNotFoundError(f"Expected wfc3-psf script at {script}")
    cmd = ["python", str(script), str(recipe_path)]
    subprocess.check_call(cmd, cwd=str(repo_path))
