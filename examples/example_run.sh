#!/bin/bash
# Example run wrapper
# Usage: ./example_run.sh /path/to/_rate.fits /path/to/_flt.fits

INPUTS="$@"
JWST_REPO="/home/user/JWST-Merge"
WFC3_REPO="/home/user/wfc3-psf"
OUT="nsm_example_out"
nsm run $INPUTS --jwst-repo $JWST_REPO --wfc3-repo $WFC3_REPO -o $OUT
