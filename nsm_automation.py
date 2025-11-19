from nsm_auto_detect import scan_directory
from nsm_pipeline_run import run_jwst_merge, run_wfc3_psf
import os
import json

def run_all(root):
    detected = scan_directory(root)
    results = {}

    # --- JWST ---
    if detected["JWST"]:
        print("Running JWST-Merge...")
        out = run_jwst_merge(detected["JWST"], "output/jwst")
        results["jwst"] = out.stdout

    # --- WFC3 ---
    if detected["WFC3"]:
        print("Running WFC3-PSF...")
        out = run_wfc3_psf(detected["WFC3"], "output/wfc3")
        results["wfc3"] = out.stdout

    # Write automation summary
    with open("output/automation_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Automation complete.")
    return results
