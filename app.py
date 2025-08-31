import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="HDB Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 HDB Resale Price Prediction (Linear Pipeline)")

# ----------- DEFAULT PATHS -----------
DEFAULT_MODEL_PATH = "models/hdb_price_pipeline.pkl"
DEFAULT_DATA_PATH  = "data/hdb_processed_data.csv"

# ----------- SIDEBAR: Uploads & Metadata -----------
st.sidebar.header("⚙️ Options")
uploaded_model = st.sidebar.file_uploader("Model pipeline (.pkl)", type=["pkl"])
uploaded_data  = st.sidebar.file_uploader("Training CSV (for feature schema)", type=["csv"])

st.sidebar.header("👤 Student Metadata")
student_name = st.sidebar.text_input("Name", value="Aung Hlaing Tun")
student_id = st.sidebar.text_input("Student ID", value="6319250G")
course = st.sidebar.text_input("Course", value="ITI-105")
team_id = st.sidebar.text_input("Project Group ID", value="AlogoRiddler")
project_date = st.sidebar.text_input("Project Date", value="25 Aug 2025")
sg_now = datetime.now(ZoneInfo("Asia/Singapore"))
inference_ts = st.sidebar.text_input("Model Inference Date (SGT)", value=sg_now.strftime("%Y-%m-%d %H:%M:%S %Z"))

# ----------- Handle Uploads -----------
MODEL_PATH = DEFAULT_MODEL_PATH
DATA_PATH  = DEFAULT_DATA_PATH

if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        tmp.write(uploaded_model.read())
        MODEL_PATH = tmp.name

if uploaded_data:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_data.read())
        DATA_PATH = tmp.name

# ----------- Helpers -----------
def clamp(val, lo, hi, fallback=None):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return fallback if fallback is not None else lo
    try:
        v = float(val)
    except Exception:
        return fallback if fallback is not None else lo
    return max(min(v, hi), lo)

def clamp_int(val, lo, hi, fallback=None):
    return int(round(clamp(val, lo, hi, fallback)))

@st.cache_resource(show_spinner=False)
def load_model(path): return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_template(path):
    df = pd.read_csv(path, nrows=1)
    X = df.drop(columns=["resale_price"]) if "resale_price" in df.columns else df.copy()
    return X, list(X.columns)

@st.cache_data(show_spinner=False)
def get_category_choices(path, available_cols, cols=("town","flat_type","flat_model","storey_range")):
    usecols = [c for c in cols if c in available_cols]
    if not usecols: return {}
    df = pd.read_csv(path, usecols=usecols)
    return {c: sorted(df[c].dropna().astype(str).unique()) for c in df.columns}

# ----------- Load Model & Template -----------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()
if not os.path.exists(DATA_PATH):
    st.error(f"Training CSV not found at {DATA_PATH}")
    st.stop()

pipe = load_model(MODEL_PATH)
X_template, required_cols = load_template(DATA_PATH)
choices = get_category_choices(DATA_PATH, required_cols)

# ----------- Metadata Pill -----------
st.markdown(
    f"""
    <div style="padding:10px;border-radius:8px;background:#F7F9FC;border:1px solid #E6EAF2;margin-bottom:8px">
      <strong>Name:</strong> {student_name} &nbsp;|&nbsp;
      <strong>ID:</strong> {student_id} &nbsp;|&nbsp;
      <strong>Course:</strong> {course} &nbsp;|&nbsp;
      <strong>Group:</strong> {team_id} &nbsp;|&nbsp;
      <strong>Project Date:</strong> {project_date} &nbsp;|&nbsp;
      <strong>Inference:</strong> {inference_ts}
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------- UI Inputs -----------
def get_num(name, default): return float(X_template[name].iloc[0]) if name in X_template else default
def get_str(name, default): return str(X_template[name].iloc[0]) if name in X_template else default

col1, col2 = st.columns(2)
with col1:
    floor_area_sqm = st.number_input("Floor Area (sqm)", 20.0, 250.0, clamp(get_num("floor_area_sqm", 90.0), 20.0, 250.0), step=1.0)
    remaining_lease_years = st.number_input("Remaining Lease (years)", 1.0, 99.0, clamp(get_num("remaining_lease_years", 70.0), 1.0, 99.0), step=1.0)
    min_storey = st.number_input("Min Storey", 1, 60, clamp_int(get_num("min_storey", 1), 1, 60), step=1)
    max_storey = st.number_input("Max Storey", 1, 60, clamp_int(get_num("max_storey", 3), 1, 60), step=1)
with col2:
    latitude = st.number_input("Latitude", 1.20, 1.50, clamp(get_num("latitude", 1.3500), 1.20, 1.50), step=0.0001, format="%.6f")
    longitude = st.number_input("Longitude", 103.60, 104.10, clamp(get_num("longitude", 103.8500), 103.60, 104.10), step=0.0001, format="%.6f")
    cpi = st.number_input("CPI", 80.0, 140.0, clamp(get_num("cpi", 105.0), 80.0, 140.0), step=0.1)
    distance_to_mrt = st.number_input("Distance to MRT (km)", 0.0, 20.0, clamp(get_num("distance_to_mrt", 1.2), 0.0, 20.0), step=0.1)

year = st.number_input("Transaction Year", 2015, 2025, clamp_int(get_num("year", 2024), 2015, 2025), step=1)
month_num = st.selectbox("Transaction Month (1–12)", list(range(1, 13)), index=clamp_int(get_num("month_num", 1), 1, 12) - 1)

if max_storey < min_storey:
    st.info("Max Storey adjusted to match Min Storey.")
    max_storey = min_storey
mid_storey_val = (min_storey + max_storey) / 2.0

def cat_input(label, col_name, default):
    opts = choices.get(col_name, [])
    return st.selectbox(label, opts) if opts else st.text_input(label, value=default)

town         = cat_input("Town", "town", get_str("town", "ANG MO KIO"))
flat_type    = cat_input("Flat Type", "flat_type", get_str("flat_type", "3 ROOM"))
flat_model   = cat_input("Flat Model", "flat_model", get_str("flat_model", "Model A"))
storey_range = cat_input("Storey Range", "storey_range", get_str("storey_range", "01 TO 03"))

# ----------- Prediction Logic -----------
show_row = st.checkbox("Show feature row sent to model", value=False)
pred_val = None

if st.button("Predict Price"):
    X_input = X_template.copy()

    def set_if_present(col, val):
        if col in X_input.columns:
            X_input[col] = val

    # Set inputs
    for col, val in {
        "floor_area_sqm": floor_area_sqm,
        "remaining_lease_years": remaining_lease_years,
        "min_storey": min_storey,
        "max_storey": max_storey,
        "mid_storey": mid_storey_val,
        "latitude": latitude,
        "longitude": longitude,
        "cpi": cpi,
        "distance_to_mrt": distance_to_mrt,
        "year": year,
        "month_num": int(month_num),
        "town": town,
        "flat_type": flat_type,
        "flat_model": flat_model,
        "storey_range": storey_range
    }.items():
        set_if_present(col, val)

    X_input = X_input[required_cols]

