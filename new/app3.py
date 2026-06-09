# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="DBSCAN Clustering App",
    layout="wide"
)

st.title("DBSCAN Clustering using Streamlit")

st.write("""
Upload a CSV dataset and perform DBSCAN clustering interactively.
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

    # Numeric columns only
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

            st.subheader("DBSCAN Settings")

            col1, col2 = st.columns(2)

            with col1:
                eps = st.slider(
                    "Epsilon (eps)",
                    min_value=0.1,
                    max_value=5.0,
                    value=0.5,
                    step=0.1
                )

            with col2:
                min_samples = st.slider(
                    "Minimum Samples",
                    min_value=2,
                    max_value=20,
                    value=5
                )

            # Train DBSCAN model
            model = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )

            labels = model.fit_predict(scaled_data)

            # Add cluster labels
            df["Cluster"] = labels

            st.subheader("Clustered Data")
            st.dataframe(df)

            # Number of clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # Noise points
            noise_points = list(labels).count(-1)

            st.success(f"Number of Clusters: {n_clusters}")
            st.warning(f"Noise Points: {noise_points}")

            # Silhouette Score
            unique_labels = set(labels)

            if len(unique_labels) > 1 and n_clusters > 1:

                score = silhouette_score(scaled_data, labels)

                st.info(f"Silhouette Score: {round(score, 4)}")

            else:
                st.error("Silhouette Score cannot be calculated")

            # Scatter Plot
            st.subheader("Cluster Visualization")

            fig, ax = plt.subplots(figsize=(8, 6))

            scatter = ax.scatter(
                data[selected_features[0]],
                data[selected_features[1]],
                c=labels
            )

            ax.set_xlabel(selected_features[0])
            ax.set_ylabel(selected_features[1])

            ax.set_title("DBSCAN Clusters")

            st.pyplot(fig)

            # Show Noise Points
            st.subheader("Understanding DBSCAN Output")

            st.write("""
- Cluster labels start from 0
- `-1` indicates noise/outlier points
- DBSCAN automatically determines number of clusters
""")

else:
    st.info("Please upload a CSV file.")