import yaml
from pathlib import Path
from .utils import read_fits_header
from .advisor import ai_suggest_params
from .runners.jwst_runner import run_jwst_merge
from .runners.wfc3_runner import run_wfc3_psf

def detect_instrument(path):
    hdr = read_fits_header(path)
    return hdr.get("INSTRUME", "").lower()

def choose_pipeline(instrument):
    if any(k in instrument for k in ("nircam", "miri")):
        return "jwst"
    if "wfc3" in instrument:
        return "wfc3"
    return "unknown"

def generate_recipe(output_path: Path, inputs, pipeline, params):
    recipe = {"inputs": [str(p) for p in inputs], "pipeline": pipeline, "params": params}
    output_path.write_text(yaml.safe_dump(recipe))
    return output_path

def orchestrate_run(input_files, jwst_repo=None, wfc3_repo=None, out_dir="nsm_output"):
    in_paths = [Path(p).resolve() for p in input_files]
    # detect first instrument (simple heuristic: use first file)
    inst = detect_instrument(in_paths[0])
    pipeline = choose_pipeline(inst)
    params = ai_suggest_params(read_fits_header(in_paths[0]), goal="science-grade")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = out_dir / "run_recipe.yml"
    generate_recipe(recipe_path, in_paths, pipeline, params)

    if pipeline == "jwst":
        if not jwst_repo:
            raise RuntimeError("jwst_repo path required to run JWST-Merge.")
        run_jwst_merge(recipe_path, Path(jwst_repo))

    elif pipeline == "wfc3":
        if not wfc3_repo:
            raise RuntimeError("wfc3_repo path required to run wfc3-psf.")
        run_wfc3_psf(recipe_path, Path(wfc3_repo))

    else:
        raise RuntimeError(f"Unknown pipeline for instrument: {inst}")

    # after successful run, produce a simple summary
    summary = {"pipeline": pipeline, "params": params, "recipe": str(recipe_path)}
    (out_dir / "nsm_summary.yml").write_text(yaml.safe_dump(summary))
    return summary
