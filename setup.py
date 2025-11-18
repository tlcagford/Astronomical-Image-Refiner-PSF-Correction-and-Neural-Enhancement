from setuptools import setup, find_packages

setup(
    name="neuro_symmetry_mapper",
    version="0.1.0",
    description="Neuro-Symmetry Mapper: orchestrator and UI for JWST-Merge and wfc3-psf",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "astropy",
        "PyYAML",
        "typer",
        "rich",
        "ipython",
        "ipywidgets",
        "voila",
        "matplotlib",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "nsm=nsm.cli:app",
        ],
    },
)
