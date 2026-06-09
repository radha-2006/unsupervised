# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="KMeans Clustering App",
    layout="wide"
)

st.title("KMeans Clustering using Streamlit")

st.write("""
Upload a CSV dataset and perform KMeans clustering interactively.
""")

# Upload file
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
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

            st.subheader("KMeans Settings")

            col1, col2 = st.columns(2)

            with col1:
                n_clusters = st.slider(
                    "Number of Clusters",
                    min_value=2,
                    max_value=10,
                    value=3
                )

            with col2:
                init_method = st.selectbox(
                    "Initialization Method",
                    ["k-means++", "random"]
                )

            # Train model
            model = KMeans(
                n_clusters=n_clusters,
                init=init_method,
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(scaled_data)

            # Add cluster labels
            df["Cluster"] = labels

            st.subheader("Clustered Data")
            st.dataframe(df)

            # Silhouette Score
            score = silhouette_score(scaled_data, labels)

            st.success(f"Silhouette Score: {round(score, 4)}")

            # Elbow Method
            st.subheader("Elbow Method")

            inertia_values = []

            cluster_range = range(1, 11)

            for k in cluster_range:

                km = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10
                )

                km.fit(scaled_data)

                inertia_values.append(km.inertia_)

            fig1, ax1 = plt.subplots(figsize=(8, 5))

            ax1.plot(cluster_range, inertia_values, marker='o')

            ax1.set_xlabel("Number of Clusters")
            ax1.set_ylabel("Inertia (WCSS)")
            ax1.set_title("Elbow Method")

            st.pyplot(fig1)

            # Scatter Plot
            st.subheader("Cluster Visualization")

            fig2, ax2 = plt.subplots(figsize=(8, 6))

            scatter = ax2.scatter(
                data[selected_features[0]],
                data[selected_features[1]],
                c=labels
            )

            # Centroids
            centroids = scaler.inverse_transform(model.cluster_centers_)

            ax2.scatter(
                centroids[:, 0],
                centroids[:, 1],
                marker='X',
                s=300
            )

            ax2.set_xlabel(selected_features[0])
            ax2.set_ylabel(selected_features[1])
            ax2.set_title("KMeans Clusters")

            st.pyplot(fig2)

else:
    st.info("Please upload a CSV file.")