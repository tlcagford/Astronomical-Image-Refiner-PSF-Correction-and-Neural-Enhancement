# notebooks/voila_dashboard.py
# %% [markdown]
# Neuro-Symmetry Mapper — Voila Dashboard
# Open in Jupyter and run; serve with: `voila notebooks/voila_dashboard.py`

# %%
import ipywidgets as widgets
from IPython.display import display, clear_output
from pathlib import Path
import yaml
from nsm.orchestrator import orchestrate_run, detect_instrument, ai_suggest_params
from nsm.utils import read_fits_header

# File picker widget
file_picker = widgets.FileUpload(accept='.fits,.fz', multiple=True)
jwst_repo_box = widgets.Text(value="", description="JWST repo path")
wfc3_repo_box = widgets.Text(value="", description="WFC3 repo path")
run_button = widgets.Button(description="Run NSM", button_style="success")
output_area = widgets.Output(layout={'border': '1px solid black'})

# Display header
display(widgets.HTML("<h2>Neuro-Symmetry Mapper — Quick Run</h2>"))
display(widgets.HTML("Upload your small example FITS files (or use local paths below)"))

# Local path input as fallback
local_paths = widgets.Text(value="", description="Local paths (comma sep)")

display(widgets.HBox([jwst_repo_box, wfc3_repo_box]))
display(local_paths)
display(file_picker)
display(run_button)
display(output_area)

# Handler
def on_run_clicked(b):
    with output_area:
        clear_output()
        # Determine inputs
        input_files = []
        if file_picker.value:
            # save uploads to a temp folder
            temp_dir = Path("nsm_uploaded")
            temp_dir.mkdir(exist_ok=True)
            for fn, info in file_picker.value.items():
                path = temp_dir / fn
                with open(path, "wb") as f:
                    f.write(info['content'])
                input_files.append(str(path))
        elif local_paths.value.strip():
            input_files = [p.strip() for p in local_paths.value.split(",") if p.strip()]
        else:
            print("No inputs provided.")
            return

        print("Detected inputs:", input_files)
        try:
            summary = orchestrate_run(input_files,
                                      jwst_repo=jwst_repo_box.value or None,
                                      wfc3_repo=wfc3_repo_box.value or None,
                                      out_dir="nsm_voila_output")
            print("Run summary:")
            print(yaml.safe_dump(summary, sort_keys=False))
            print("Artifacts written to nsm_voila_output/")
        except Exception as e:
            print("Error during run:", e)

run_button.on_click(on_run_clicked)
