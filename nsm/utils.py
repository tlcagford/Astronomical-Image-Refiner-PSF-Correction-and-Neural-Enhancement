from astropy.io import fits

from pathlib import Path

def read_fits_header(path):
    hdr = fits.getheader(str(path))
    return dict(hdr)

def safe_path(p):
    return Path(p).expanduser().resolve()
