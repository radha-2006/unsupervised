# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="PCA Dimensionality Reduction",
    layout="wide"
)

st.title("PCA using Streamlit")

st.write("""
Upload a CSV dataset and perform Principal Component Analysis (PCA).
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
            default=numeric_cols
        )

        if len(selected_features) >= 2:

            data = df[selected_features]

            # Scaling
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            st.subheader("PCA Settings")

            n_components = st.slider(
                "Number of Principal Components",
                min_value=2,
                max_value=min(len(selected_features), 10),
                value=2
            )

            # PCA Model
            pca = PCA(n_components=n_components)

            principal_components = pca.fit_transform(scaled_data)

            # Create PCA DataFrame
            pca_columns = [
                f"PC{i+1}"
                for i in range(n_components)
            ]

            pca_df = pd.DataFrame(
                principal_components,
                columns=pca_columns
            )

            st.subheader("Transformed PCA Data")
            st.dataframe(pca_df)

            # Explained Variance Ratio
            st.subheader("Explained Variance Ratio")

            variance_df = pd.DataFrame({
                "Principal Component": pca_columns,
                "Explained Variance": pca.explained_variance_ratio_
            })

            st.dataframe(variance_df)

            # Variance Plot
            fig1, ax1 = plt.subplots(figsize=(8, 5))

            ax1.bar(
                pca_columns,
                pca.explained_variance_ratio_
            )

            ax1.set_xlabel("Principal Components")
            ax1.set_ylabel("Explained Variance Ratio")
            ax1.set_title("Variance Explained by Components")

            st.pyplot(fig1)

            # Cumulative Variance
            st.subheader("Cumulative Explained Variance")

            cumulative_variance = np.cumsum(
                pca.explained_variance_ratio_
            )

            fig2, ax2 = plt.subplots(figsize=(8, 5))

            ax2.plot(
                range(1, n_components + 1),
                cumulative_variance,
                marker='o'
            )

            ax2.set_xlabel("Number of Components")
            ax2.set_ylabel("Cumulative Variance")
            ax2.set_title("Cumulative Explained Variance")

            st.pyplot(fig2)

            # PCA Scatter Plot
            if n_components >= 2:

                st.subheader("PCA Scatter Plot")

                fig3, ax3 = plt.subplots(figsize=(8, 6))

                ax3.scatter(
                    principal_components[:, 0],
                    principal_components[:, 1]
                )

                ax3.set_xlabel("Principal Component 1")
                ax3.set_ylabel("Principal Component 2")

                ax3.set_title("PCA Projection")

                st.pyplot(fig3)

            # Feature Importance
            st.subheader("PCA Component Weights")

            component_df = pd.DataFrame(
                pca.components_,
                columns=selected_features,
                index=pca_columns
            )

            st.dataframe(component_df)

            st.write("""
Higher absolute values indicate stronger contribution
of a feature to that principal component.
""")

else:
    st.info("Please upload a CSV file.")