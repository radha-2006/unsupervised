# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Isolation Forest Anomaly Detection",
    layout="wide"
)

st.title("Isolation Forest using Streamlit")

st.write("""
Upload a CSV dataset and detect anomalies using Isolation Forest.
""")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns")

    else:

        # Feature Selection
        selected_features = st.multiselect(
            "Select Features",
            numeric_cols,
            default=numeric_cols[:2]
        )

        if len(selected_features) >= 2:

            data = df[selected_features]

            # Scaling
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            st.subheader("Isolation Forest Settings")

            col1, col2 = st.columns(2)

            with col1:
                contamination = st.slider(
                    "Contamination Ratio",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.05,
                    step=0.01
                )

            with col2:
                n_estimators = st.slider(
                    "Number of Trees",
                    min_value=50,
                    max_value=500,
                    value=100,
                    step=50
                )

            # Train Isolation Forest
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=42
            )

            model.fit(scaled_data)

            # Predictions
            predictions = model.predict(scaled_data)

            # Convert:
            # -1 -> anomaly
            #  1 -> normal
            anomaly_labels = np.where(predictions == -1, 1, 0)

            df["Anomaly"] = anomaly_labels

            st.subheader("Anomaly Detection Results")
            st.dataframe(df)

            # Count anomalies
            anomaly_count = anomaly_labels.sum()

            st.error(f"Detected Anomalies: {anomaly_count}")

            # Anomaly Scores
            scores = model.decision_function(scaled_data)

            df["Anomaly_Score"] = scores

            st.subheader("Anomaly Scores")
            st.dataframe(
                df[["Anomaly", "Anomaly_Score"]]
            )

            # Visualization
            st.subheader("Anomaly Visualization")

            fig, ax = plt.subplots(figsize=(8, 6))

            normal_points = df[df["Anomaly"] == 0]
            anomaly_points = df[df["Anomaly"] == 1]

            ax.scatter(
                normal_points[selected_features[0]],
                normal_points[selected_features[1]],
                label="Normal"
            )

            ax.scatter(
                anomaly_points[selected_features[0]],
                anomaly_points[selected_features[1]],
                label="Anomaly"
            )

            ax.set_xlabel(selected_features[0])
            ax.set_ylabel(selected_features[1])

            ax.set_title("Isolation Forest Anomaly Detection")

            ax.legend()

            st.pyplot(fig)

            # Explain anomaly score
            st.subheader("Understanding Isolation Forest")

            st.write("""
Isolation Forest isolates anomalies instead of profiling normal data.

Key intuition:
- anomalies are rare
- anomalies are easier to isolate

Hence:
- fewer splits needed
- shorter path length
- higher anomaly probability
""")

else:
    st.info("Please upload a CSV file.")