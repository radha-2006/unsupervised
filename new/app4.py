# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Gaussian Mixture Model App",
    layout="wide"
)

st.title("Gaussian Mixture Model (GMM) Clustering")

st.write("""
Upload a CSV dataset and perform clustering using Gaussian Mixture Models.
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

        # Feature selection
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

            st.subheader("GMM Settings")

            col1, col2 = st.columns(2)

            with col1:
                n_components = st.slider(
                    "Number of Components",
                    min_value=2,
                    max_value=10,
                    value=3
                )

            with col2:
                covariance_type = st.selectbox(
                    "Covariance Type",
                    ["full", "tied", "diag", "spherical"]
                )

            # Train GMM Model
            model = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                random_state=42
            )

            model.fit(scaled_data)

            # Predict clusters
            labels = model.predict(scaled_data)

            # Probability scores
            probabilities = model.predict_proba(scaled_data)

            # Add cluster labels
            df["Cluster"] = labels

            st.subheader("Clustered Data")
            st.dataframe(df)

            # Silhouette Score
            score = silhouette_score(scaled_data, labels)

            st.success(f"Silhouette Score: {round(score, 4)}")

            # AIC and BIC
            aic = model.aic(scaled_data)
            bic = model.bic(scaled_data)

            col3, col4 = st.columns(2)

            with col3:
                st.info(f"AIC Score: {round(aic, 2)}")

            with col4:
                st.info(f"BIC Score: {round(bic, 2)}")

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

            ax.set_title("GMM Clusters")

            st.pyplot(fig)

            # Probability Output
            st.subheader("Cluster Membership Probabilities")

            prob_df = pd.DataFrame(
                probabilities,
                columns=[
                    f"Cluster_{i}_Probability"
                    for i in range(n_components)
                ]
            )

            st.dataframe(prob_df.head())

            # Explain Soft Clustering
            st.subheader("Soft Clustering")

            st.write("""
Unlike KMeans:
- KMeans assigns one fixed cluster to each point
- GMM gives probability of belonging to every cluster

Example:
- Point A:
    - Cluster 1 → 80%
    - Cluster 2 → 20%

This is called probabilistic clustering.
""")

else:
    st.info("Please upload a CSV file.")