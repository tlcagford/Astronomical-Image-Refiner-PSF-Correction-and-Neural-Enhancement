python -c "from nsm.cli import app; print('installed')"
# or run orchestrator directly:
python -c "from nsm.orchestrator import orchestrate_run; orchestrate_run(['examples/foo_rate.fits'], jwst_repo='/path/to/JWST-Merge')"
