# Human Patch Decision Trace Web App

This is a small Streamlit webpage for collecting a human decision trace from microscopy image patches.

## What it does

- Loads an AFM `.ibw` file directly (or PNG/JPG/TIFF/NPY for testing).
- Lets you choose the channel; for a 6-channel DART file it uses the same names as the earlier notebook.
- Globally min-max normalizes the selected image to `[0, 1]`.
- Shows **20 random patches per step** by default.
- Runs for **100 decision steps** by default.
- Records the clicked candidate at every step.
- Saves an analysis-ready Python pickle with the same core per-step fields used previously:
  - `patches`
  - `selected_candidate`
  - `selected_patch`
  - `position`
- Also saves `candidate_positions` for all candidates, plus timestamps and step number.
- Provides a compact CSV summary and JSON metadata file at the end.

## Run locally

Open Terminal in this folder and run:

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Output PKL structure

The downloaded `human_decision_trace.pkl` is a dictionary keyed by zero-based step number:

```python
trace[0] = {
    "patches": ...,                 # shape: (20, patch_size, patch_size)
    "candidate_positions": ...,     # [(x0, y0), ..., (x19, y19)]
    "selected_candidate": 7,
    "selected_patch": ...,
    "position": (x, y),             # selected candidate center
    "step": 0,
    "selection_timestamp_utc": "...",
}
```

So the existing analysis code can continue to use:

```python
with open("human_decision_trace.pkl", "rb") as f:
    trace = pickle.load(f)

step_data = trace[0]
patches = step_data["patches"]
selected = step_data["selected_candidate"]
selected_patch = step_data["selected_patch"]
position = step_data["position"]
```

## Notes

- Candidate centers are sampled without replacement **within each step**.
- A new random candidate set is generated after every selection.
- The random seed is saved in metadata for reproducibility.
- Candidate and step indices are zero-based to match the earlier analysis code.
