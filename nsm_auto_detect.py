import os
from astropy.io import fits

def detect_instrument(file_path):
    """Return JWST, HST-WFC3, or UNKNOWN."""
    try:
        hdr = fits.getheader(file_path)
        mission = hdr.get("TELESCOP")
        instr = hdr.get("INSTRUME")

        if mission == "JWST":
            return "JWST"
        if mission == "HST" and "WFC3" in instr:
            return "WFC3"
        return "UNKNOWN"

    except Exception:
        return "UNKNOWN"


def scan_directory(root):
    """Locate all FITS files and classify by instrument."""
    results = {"JWST": [], "WFC3": [], "UNKNOWN": []}

    for path, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".fits"):
                full = os.path.join(path, f)
                tag = detect_instrument(full)
                results[tag].append(full)

    return results
