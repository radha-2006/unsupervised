# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Hierarchical Clustering App", layout="wide")

st.title("Hierarchical Clustering using Streamlit")

st.write("""
Upload a CSV file and perform Hierarchical Clustering interactively.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Read Data
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.dataframe(df)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns")
    else:

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

            st.subheader("Clustering Settings")

            col1, col2 = st.columns(2)

            with col1:
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=10,
                    value=3
                )

            with col2:
                linkage_method = st.selectbox(
                    "Linkage Method",
                    ["ward", "complete", "average", "single"]
                )

            # Dendrogram
            st.subheader("Dendrogram")

            linked = linkage(scaled_data, method=linkage_method)

            fig, ax = plt.subplots(figsize=(10, 5))

            dendrogram(linked, ax=ax)

            plt.xlabel("Data Points")
            plt.ylabel("Distance")

            st.pyplot(fig)

            # Model
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method
            )

            labels = model.fit_predict(scaled_data)

            # Add cluster labels
            df["Cluster"] = labels

            st.subheader("Clustered Data")
            st.dataframe(df)

            # Silhouette Score
            score = silhouette_score(scaled_data, labels)

            st.success(f"Silhouette Score: {round(score, 4)}")

            # Scatter Plot
            if len(selected_features) >= 2:

                st.subheader("Cluster Visualization")

                fig2, ax2 = plt.subplots(figsize=(8, 6))

                scatter = ax2.scatter(
                    data[selected_features[0]],
                    data[selected_features[1]],
                    c=labels
                )

                ax2.set_xlabel(selected_features[0])
                ax2.set_ylabel(selected_features[1])

                st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to continue.")