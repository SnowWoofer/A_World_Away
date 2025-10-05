import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Exoplanet Classifier", layout="wide")
st.title("K2 Exoplanet Classifier 🌌")

MODEL_PATH = "../model/rain_k2_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Current working directory: {os.getcwd()}")
else:
    model = joblib.load(MODEL_PATH)

NUMERIC_COLS = ['pl_orbper', 'pl_rade', 'pl_eqt', 'st_teff', 'pl_bmasse']
CATEGORICAL_COLS = ['discoverymethod']

uploaded_file = st.file_uploader("Upload a CSV file with exoplanet data", type=["csv"])

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file, comment="#")
        st.write("File uploaded successfully!")
        
        missing_cols = [c for c in NUMERIC_COLS + CATEGORICAL_COLS if c not in df_upload.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            for col in NUMERIC_COLS:
                df_upload[col] = df_upload[col].fillna(df_upload[col].median())

            df_encoded = pd.get_dummies(df_upload, columns=CATEGORICAL_COLS)

            model_features = model.feature_names_in_
            for col in model_features:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0

            X = df_encoded[model_features]

            predictions = model.predict(X)
            df_upload["PredictedDisposition"] = predictions

            st.success("Prediction complete!")
            st.dataframe(df_upload)
            st.bar_chart(df_upload["PredictedDisposition"].value_counts())

    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
    except pd.errors.ParserError:
        st.error("The uploaded file could not be parsed. Make sure it’s a valid CSV.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
