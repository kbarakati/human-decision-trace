import io
import json
import pickle
import tempfile
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from PIL import Image
from igor2 import binarywave


# ============================================================
# CONFIGURATION
# ============================================================
DEFAULT_N_CANDIDATES = 20
DEFAULT_N_STEPS = 100
DEFAULT_PATCH_SIZE = 16
DEFAULT_SEED = 42

st.set_page_config(
    page_title="Human Patch Decision Trace",
    page_icon="🔬",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================
def normalize_01(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D image to [0, 1]."""
    image = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(image)

    if not np.any(finite):
        return np.zeros_like(image, dtype=np.float32)

    lo = np.nanmin(image[finite])
    hi = np.nanmax(image[finite])

    if hi <= lo:
        out = np.zeros_like(image, dtype=np.float32)
        out[finite] = 0.0
        return out

    out = (image - lo) / (hi - lo)
    out[~finite] = 0.0
    return out.astype(np.float32)


def load_uploaded_file(uploaded_file):
    """
    Return:
        images: np.ndarray with shape (n_channels, height, width)
        channel_names: list[str]
        source_kind: str
    """
    suffix = uploaded_file.name.lower().split(".")[-1]
    raw = uploaded_file.getvalue()

    if suffix == "ibw":
        # igor2.binarywave.load expects a path-like file.
        with tempfile.NamedTemporaryFile(suffix=".ibw") as tmp:
            tmp.write(raw)
            tmp.flush()
            ibw = binarywave.load(tmp.name)

        wave = ibw["wave"]
        data = np.asarray(wave["wData"])

        # Match the notebook convention when channels are stored last.
        # Typical DART file: (H, W, C) -> (C, H, W)
        if data.ndim == 3:
            if data.shape[-1] <= 32:
                images = np.moveaxis(data, -1, 0)
            elif data.shape[0] <= 32:
                images = data
            else:
                raise ValueError(
                    f"Could not identify the channel axis automatically. IBW shape: {data.shape}"
                )
        elif data.ndim == 2:
            images = data[None, ...]
        else:
            raise ValueError(f"Unsupported IBW data shape: {data.shape}")

        # These are the channel names used in the prior DART notebook.
        dart_names = [
            "Height",
            "Amplitude1",
            "Amplitude2",
            "Phase1",
            "Phase2",
            "Frequency",
        ]
        if images.shape[0] == len(dart_names):
            channel_names = dart_names
        else:
            channel_names = [f"Channel {i}" for i in range(images.shape[0])]

        return images.astype(np.float32), channel_names, "IBW"

    if suffix in {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}:
        image = Image.open(io.BytesIO(raw)).convert("L")
        images = np.asarray(image, dtype=np.float32)[None, ...]
        return images, ["Grayscale"], "image"

    if suffix == "npy":
        data = np.load(io.BytesIO(raw), allow_pickle=False)
        if data.ndim == 2:
            images = data[None, ...]
        elif data.ndim == 3:
            # Prefer channel-first if first axis is small; otherwise channel-last.
            if data.shape[0] <= 32:
                images = data
            elif data.shape[-1] <= 32:
                images = np.moveaxis(data, -1, 0)
            else:
                raise ValueError(f"Could not identify channel axis for NPY shape {data.shape}")
        else:
            raise ValueError(f"Unsupported NPY shape: {data.shape}")
        return images.astype(np.float32), [f"Channel {i}" for i in range(images.shape[0])], "NPY"

    raise ValueError("Supported files: .ibw, .png, .jpg, .jpeg, .tif, .tiff, .bmp, .npy")


def generate_candidates(image, n_candidates, patch_size, rng):
    """Generate unique random valid patch centers and extract patches."""
    h, w = image.shape
    half = patch_size // 2

    # For a center (x, y), patch is [y-half:y-half+size, x-half:x-half+size].
    x_min = half
    y_min = half
    x_max = w - (patch_size - half)
    y_max = h - (patch_size - half)

    if x_max < x_min or y_max < y_min:
        raise ValueError(
            f"Patch size {patch_size} is too large for image shape {image.shape}."
        )

    xs = np.arange(x_min, x_max + 1)
    ys = np.arange(y_min, y_max + 1)
    n_possible = len(xs) * len(ys)

    if n_candidates > n_possible:
        raise ValueError(
            f"Requested {n_candidates} candidates, but only {n_possible} valid centers exist."
        )

    flat_ids = rng.choice(n_possible, size=n_candidates, replace=False)
    positions = []
    patches = []

    nx = len(xs)
    for flat_id in flat_ids:
        yi = int(flat_id // nx)
        xi = int(flat_id % nx)
        x = int(xs[xi])
        y = int(ys[yi])

        x0 = x - half
        y0 = y - half
        patch = image[y0:y0 + patch_size, x0:x0 + patch_size]

        positions.append((x, y))
        patches.append(np.asarray(patch, dtype=np.float32))

    return np.stack(patches, axis=0), positions


def reset_experiment():
    for key in [
        "trace",
        "current_step",
        "current_patches",
        "current_positions",
        "rng",
        "experiment_started",
        "experiment_finished",
        "experiment_config",
        "experiment_start_time",
    ]:
        st.session_state.pop(key, None)


def initialize_experiment(image, source_name, channel_index, channel_name,
                          n_candidates, n_steps, patch_size, seed):
    st.session_state.trace = {}
    st.session_state.current_step = 0
    st.session_state.rng = np.random.default_rng(seed)
    st.session_state.experiment_started = True
    st.session_state.experiment_finished = False
    st.session_state.experiment_start_time = datetime.now(timezone.utc).isoformat()
    st.session_state.experiment_config = {
        "source_file": source_name,
        "channel_index": int(channel_index),
        "channel_name": channel_name,
        "image_shape": [int(v) for v in image.shape],
        "n_candidates": int(n_candidates),
        "n_steps": int(n_steps),
        "patch_size": int(patch_size),
        "seed": int(seed),
        "image_normalization": "global min-max to [0, 1]",
        "position_order": "(x, y) pixels in normalized source image",
        "trace_indexing": "0-based step keys and 0-based candidate indices",
        "started_utc": st.session_state.experiment_start_time,
    }
    prepare_step(image)


def prepare_step(image):
    cfg = st.session_state.experiment_config
    patches, positions = generate_candidates(
        image=image,
        n_candidates=cfg["n_candidates"],
        patch_size=cfg["patch_size"],
        rng=st.session_state.rng,
    )
    st.session_state.current_patches = patches
    st.session_state.current_positions = positions


def record_choice(candidate_index, image):
    step = int(st.session_state.current_step)
    patches = st.session_state.current_patches
    positions = st.session_state.current_positions

    # Keep the same fields used by the previous decision traces.
    # Add candidate_positions and step for complete reconstruction.
    st.session_state.trace[step] = {
        "patches": np.asarray(patches, dtype=np.float32),
        "candidate_positions": [(int(x), int(y)) for x, y in positions],
        "selected_candidate": int(candidate_index),
        "selected_patch": np.asarray(patches[candidate_index], dtype=np.float32),
        "position": tuple(int(v) for v in positions[candidate_index]),
        "step": step,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    st.session_state.current_step += 1

    if st.session_state.current_step >= st.session_state.experiment_config["n_steps"]:
        st.session_state.experiment_finished = True
        st.session_state.experiment_config["finished_utc"] = datetime.now(timezone.utc).isoformat()
        st.session_state.current_patches = None
        st.session_state.current_positions = None
    else:
        prepare_step(image)


def trace_to_summary_csv(trace):
    lines = ["step,selected_candidate,x,y"]
    for step in sorted(trace.keys()):
        item = trace[step]
        x, y = item["position"]
        lines.append(f"{step},{item['selected_candidate']},{x},{y}")
    return "\n".join(lines).encode("utf-8")


def metadata_json_bytes(config, trace):
    data = dict(config)
    data["completed_steps"] = len(trace)
    data["selected_positions"] = {
        str(step): [int(v) for v in trace[step]["position"]]
        for step in sorted(trace.keys())
    }
    data["selected_candidates"] = {
        str(step): int(trace[step]["selected_candidate"])
        for step in sorted(trace.keys())
    }
    return json.dumps(data, indent=2).encode("utf-8")


# ============================================================
# PAGE
# ============================================================
st.title("Human Patch Decision Trace")
st.caption(
    "At each step, inspect 20 randomly sampled microscopy patches and click the one you prefer. "
    "The app records a complete decision trace compatible with the previous patch-choice analysis."
)

uploaded = st.file_uploader(
    "Upload the source microscopy image",
    type=["ibw", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "npy"],
    help="Use the original .ibw file for the same AFM workflow, or an image/NPY for testing.",
)

if uploaded is None:
    st.info("Upload a microscopy file to begin.")
    st.stop()

# Reset state if a different source file is uploaded.
file_signature = (uploaded.name, len(uploaded.getvalue()))
if st.session_state.get("file_signature") != file_signature:
    reset_experiment()
    st.session_state.file_signature = file_signature

try:
    images, channel_names, source_kind = load_uploaded_file(uploaded)
except Exception as exc:
    st.error(f"Could not load file: {exc}")
    st.stop()

channel_index = st.selectbox(
    "Channel",
    options=list(range(images.shape[0])),
    index=min(2, images.shape[0] - 1),
    format_func=lambda i: f"{i}: {channel_names[i]}",
    disabled=st.session_state.get("experiment_started", False),
)

image = normalize_01(images[channel_index].T if source_kind == "IBW" else images[channel_index])

with st.sidebar:
    st.header("Experiment settings")
    n_candidates = st.number_input(
        "Candidates per step",
        min_value=2,
        max_value=50,
        value=DEFAULT_N_CANDIDATES,
        step=1,
        disabled=st.session_state.get("experiment_started", False),
    )
    n_steps = st.number_input(
        "Number of steps",
        min_value=1,
        max_value=1000,
        value=DEFAULT_N_STEPS,
        step=1,
        disabled=st.session_state.get("experiment_started", False),
    )
    patch_size = st.number_input(
        "Patch size (pixels)",
        min_value=4,
        max_value=min(128, int(min(image.shape))),
        value=min(DEFAULT_PATCH_SIZE, int(min(image.shape))),
        step=1,
        disabled=st.session_state.get("experiment_started", False),
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2_147_483_647,
        value=DEFAULT_SEED,
        step=1,
        disabled=st.session_state.get("experiment_started", False),
    )

    if not st.session_state.get("experiment_started", False):
        if st.button("Start experiment", type="primary", use_container_width=True):
            initialize_experiment(
                image=image,
                source_name=uploaded.name,
                channel_index=channel_index,
                channel_name=channel_names[channel_index],
                n_candidates=int(n_candidates),
                n_steps=int(n_steps),
                patch_size=int(patch_size),
                seed=int(seed),
            )
            st.rerun()
    else:
        if st.button("Reset experiment", use_container_width=True):
            reset_experiment()
            st.rerun()

# Preview source before experiment starts.
if not st.session_state.get("experiment_started", False):
    st.subheader("Source image")
    st.image(image, clamp=True, caption=f"{uploaded.name} — {channel_names[channel_index]}")
    st.write(
        f"Image shape: **{image.shape[1]} × {image.shape[0]} px**  |  "
        f"Candidates/step: **{int(n_candidates)}**  |  Steps: **{int(n_steps)}**  |  "
        f"Patch: **{int(patch_size)} × {int(patch_size)} px**"
    )
    st.stop()

cfg = st.session_state.experiment_config
completed = len(st.session_state.trace)
progress = completed / cfg["n_steps"]
st.progress(progress, text=f"Completed {completed} / {cfg['n_steps']} decisions")

if not st.session_state.experiment_finished:
    step = st.session_state.current_step
    st.subheader(f"Step {step + 1} of {cfg['n_steps']}")
    st.write("Click **one** patch. Your choice is recorded immediately and the next random set appears.")

    patches = st.session_state.current_patches
    ncols = 5

    for row_start in range(0, cfg["n_candidates"], ncols):
        cols = st.columns(ncols)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= cfg["n_candidates"]:
                continue
            with col:
                # Upscale only for display; stored patch remains the numeric array.
                st.image(
                    patches[idx],
                    clamp=True,
                    width=150,
                    caption=f"Candidate {idx}",
                )
                if st.button(
                    f"Select {idx}",
                    key=f"select_step_{step}_candidate_{idx}",
                    use_container_width=True,
                ):
                    record_choice(idx, image)
                    st.rerun()

else:
    st.success(f"Trace complete: {len(st.session_state.trace)} decisions recorded.")
    st.subheader("Save trace")
    st.write(
        "The PKL is the analysis-ready trace. Each step uses the same core fields as the previous traces: "
        "`patches`, `selected_candidate`, `selected_patch`, and `position`. "
        "It additionally stores `candidate_positions` for all 20 options."
    )

    pkl_bytes = pickle.dumps(st.session_state.trace, protocol=pickle.HIGHEST_PROTOCOL)
    csv_bytes = trace_to_summary_csv(st.session_state.trace)
    json_bytes = metadata_json_bytes(st.session_state.experiment_config, st.session_state.trace)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download trace (.pkl)",
            data=pkl_bytes,
            file_name="human_decision_trace.pkl",
            mime="application/octet-stream",
            type="primary",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download summary (.csv)",
            data=csv_bytes,
            file_name="human_decision_trace_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "Download metadata (.json)",
            data=json_bytes,
            file_name="human_decision_trace_metadata.json",
            mime="application/json",
            use_container_width=True,
        )

    st.subheader("Recorded selections")
    summary_rows = []
    for step in sorted(st.session_state.trace.keys()):
        item = st.session_state.trace[step]
        x, y = item["position"]
        summary_rows.append(
            {
                "step": step,
                "selected_candidate": item["selected_candidate"],
                "x": x,
                "y": y,
            }
        )
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)
