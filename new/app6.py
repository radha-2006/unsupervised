# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="t-SNE Visualization App",
    layout="wide"
)

st.title("t-SNE Dimensionality Reduction using Streamlit")

st.write("""
Upload a CSV dataset and visualize high-dimensional data using t-SNE.
""")

# Upload CSV
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
            default=numeric_cols
        )

        if len(selected_features) >= 2:

            data = df[selected_features]

            # Scaling
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)

            st.subheader("t-SNE Settings")

            col1, col2 = st.columns(2)

            with col1:
                perplexity = st.slider(
                    "Perplexity",
                    min_value=5,
                    max_value=50,
                    value=30
                )

            with col2:
                learning_rate = st.slider(
                    "Learning Rate",
                    min_value=10,
                    max_value=1000,
                    value=200
                )

            col3, col4 = st.columns(2)

            with col3:
                n_iter = st.slider(
                    "Iterations",
                    min_value=250,
                    max_value=5000,
                    value=1000,
                    step=250
                )

            with col4:
                dimensions = st.selectbox(
                    "Output Dimensions",
                    [2, 3]
                )

            # t-SNE Model
            tsne = TSNE(
                n_components=dimensions,
                perplexity=perplexity,
                learning_rate=learning_rate,
                max_iter=n_iter,
                random_state=42
            )

            transformed_data = tsne.fit_transform(scaled_data)

            # Create DataFrame
            tsne_columns = [
                f"tSNE_{i+1}"
                for i in range(dimensions)
            ]

            tsne_df = pd.DataFrame(
                transformed_data,
                columns=tsne_columns
            )

            st.subheader("t-SNE Transformed Data")
            st.dataframe(tsne_df)

            # Visualization
            st.subheader("t-SNE Visualization")

            if dimensions == 2:

                fig, ax = plt.subplots(figsize=(8, 6))

                ax.scatter(
                    transformed_data[:, 0],
                    transformed_data[:, 1]
                )

                ax.set_xlabel("t-SNE Component 1")
                ax.set_ylabel("t-SNE Component 2")

                ax.set_title("2D t-SNE Projection")

                st.pyplot(fig)

            else:

                fig = plt.figure(figsize=(8, 6))

                ax = fig.add_subplot(111, projection='3d')

                ax.scatter(
                    transformed_data[:, 0],
                    transformed_data[:, 1],
                    transformed_data[:, 2]
                )

                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")
                ax.set_zlabel("t-SNE 3")

                ax.set_title("3D t-SNE Projection")

                st.pyplot(fig)

            # KL Divergence
            st.subheader("KL Divergence")

            st.success(
                f"KL Divergence after optimization: {round(tsne.kl_divergence_, 4)}"
            )

            # Explanation
            st.subheader("Understanding t-SNE")

            st.write("""
t-SNE tries to preserve local neighborhood relationships.

Points close in high-dimensional space:
- remain close in lower dimensions

Points far apart:
- may not preserve exact distances

This makes t-SNE excellent for:
- visualization
- cluster discovery
- embedding analysis
""")

else:
    st.info("Please upload a CSV file.")