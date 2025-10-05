import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="K2 Exoplanet Classifier 🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("K2 Exoplanet Classifier 🌌")
st.markdown(
    "Upload your K2 exoplanet data CSV to predict dispositions: `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`."
)

MODEL_PATH = "model/rain_k2_model.pkl"  # Adjust relative to src/
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Current working directory: {os.getcwd()}")
    st.stop()
else:
    model = joblib.load(MODEL_PATH)

NUMERIC_COLS = ['pl_orbper', 'pl_rade', 'pl_eqt', 'st_teff', 'pl_bmasse']
CATEGORICAL_COLS = ['discoverymethod']

pred_map = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

uploaded_file = st.file_uploader("Upload a CSV file with exoplanet data", type=["csv"])

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file, comment="#")
        st.success("File uploaded successfully!")
        st.subheader("Uploaded Data Preview")
        st.dataframe(df_upload.head())

        missing_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c not in df_upload.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            for col in NUMERIC_COLS:
                df_upload[col] = df_upload[col].fillna(df_upload[col].median())

            df_encoded = pd.get_dummies(df_upload, columns=CATEGORICAL_COLS)

            for col in model.feature_names_in_:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0

            X = df_encoded[model.feature_names_in_]

            predictions = model.predict(X)
            df_upload["PredictedDisposition"] = [pred_map[p] for p in predictions]

            cols = df_upload.columns.tolist()
            insert_pos = 1
            if 'host_name' in cols:
                insert_pos = cols.index('host_name') + 1
            cols.remove("PredictedDisposition")
            cols.insert(insert_pos, "PredictedDisposition")
            df_upload = df_upload[cols]

            st.subheader("Prediction Results")
            st.dataframe(df_upload)

            st.subheader("Prediction Distribution")
            counts = df_upload["PredictedDisposition"].value_counts().sort_index()
            st.line_chart(counts)

            st.subheader("Prediction Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Planets", len(df_upload))
            col2.metric("Confirmed", counts.get("CONFIRMED", 0))
            col3.metric("Candidates", counts.get("CANDIDATE", 0))
            st.metric("False Positives", counts.get("FALSE POSITIVE", 0))

    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
    except pd.errors.ParserError:
        st.error("The uploaded file could not be parsed. Make sure it’s a valid CSV.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
