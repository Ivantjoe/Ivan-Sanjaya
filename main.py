import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# Title and description
st.title("Credit Card Customer Clustering")
st.markdown("""
Aplikasi ini menggunakan algoritma K-Means untuk mengelompokkan data pelanggan kartu kredit.
Hasil clustering divisualisasikan untuk memahami pola kelompok.
""")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Credit Card Customer Data (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset", data)

    # Dataset statistics and checks
    st.write("### Data Description")
    st.write(data.describe())

    st.write("### Missing Values")
    st.write(data.isnull().sum())

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Select features for clustering
    st.write("### Select Features for Clustering")
    features = st.multiselect("Choose Features", data.columns, default=[data.columns[2], data.columns[3]])

    if len(features) >= 2:
        x = data[features]
        st.write("### Selected Data", x)

        # Scaling the data
        sc = StandardScaler()
        x_scaled = sc.fit_transform(x)

        # Elbow Method for optimal clusters
        wcss = []
        for i in range(1, 10):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
            kmeans.fit(x_scaled)
            wcss.append(kmeans.inertia_)

        st.write("### The Elbow Method")
        fig, ax = plt.subplots()
        ax.plot(range(1, 10), wcss, marker='o', color='red')
        ax.set_title('The Elbow Method')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)

        # User selects number of clusters
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
        pred = kmeans.fit_predict(x_scaled)
        data['Cluster'] = pred

        # Display cluster results
        st.write("### Cluster Counts")
        cluster_counts = data['Cluster'].value_counts()
        for cluster, count in cluster_counts.items():
            st.write(f"Cluster {cluster}: {count} data points")

        st.write("### Silhouette Score")
        silhouette_avg = silhouette_score(x_scaled, pred)
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")

        # Visualize clusters
        st.write("### Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red', 'purple', 'green', 'blue', 'orange', 'cyan', 'pink', 'brown', 'gray']
        for i in range(n_clusters):
            plt.scatter(
                x_scaled[pred == i, 0],
                x_scaled[pred == i, 1],
                s=50,
                c=colors[i],
                label=f'Cluster {i+1}'
            )
        plt.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            c='black',
            marker='*',
            label='Cluster Centers'
        )
        plt.title('Clusters of Credit Card Customer')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.legend(loc='best')
        st.pyplot(fig)

        # Display clustered data
        st.write("### Clustered Data")
        st.write(data)
