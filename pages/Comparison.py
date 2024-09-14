import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan
import scipy.cluster.hierarchy as sch
import altair as alt

st.set_page_config(page_title="Clustering Dashboard", layout="wide")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
alt.themes.enable("dark")

st.title('Comparison of Clustering Algorithms')

# Load datasets
minmax_ica = pd.read_csv('MinMax_ICA.csv')
robust_ica = pd.read_csv('Robust_ICA.csv')
zscore_ica = pd.read_csv('Zscore_ICA.csv')

algo_dataset_map = {
    'Hierarchical Clustering': minmax_ica,
    'HDBSCAN': zscore_ica,
    'Spectral Clustering': minmax_ica,
    'Ward Clustering': minmax_ica,
    'Birch Clustering': minmax_ica,
    'Gaussian Mixture Models (GMM)': minmax_ica
}

algo1 = st.sidebar.selectbox(
    'Select Algorithm 1',
    ['Hierarchical Clustering',
     'HDBSCAN',
     'Spectral Clustering',
     'Ward Clustering',
     'Birch Clustering',
     'Gaussian Mixture Models (GMM)'],
    index=0
)

algo2 = st.sidebar.selectbox(
    'Select Algorithm 2',
    ['Hierarchical Clustering',
     'HDBSCAN',
     'Spectral Clustering',
     'Ward Clustering',
     'Birch Clustering',
     'Gaussian Mixture Models (GMM)'],
    index=1
)

params = {
    'Hierarchical Clustering': {'n_clusters': 3, 'linkage': 'average', 'metrics': 'cityblock'},
    'HDBSCAN': {'min_cluster_size': 20, 'min_samples': 5},
    'Spectral Clustering': {'n_clusters': 3, 'affinity': 'rbf', 'n_neighbors': 5},
    'Ward Clustering': {'n_clusters': 3, 'linkage': 'average'},
    'Birch Clustering': {'n_clusters': 3, 'threshold': 0.05, 'branching_factor': 20},
    'Gaussian Mixture Models (GMM)': {'n_components': 3, 'covariance_type': 'diag', 'reg_covar': 0.00000001}
}


left_col, right_col = st.columns(2)

def perform_clustering(algorithm, params, col):
    df = algo_dataset_map[algorithm]
    params_current = params[algorithm]

    formatted_params = ', '.join([f"{key}: {value}" for key, value in params_current.items()])
    col.write(f"**Parameters:** [{formatted_params}]")

    df_features = df.drop(columns=['Cluster'], errors='ignore')  

    if algorithm == 'Hierarchical Clustering':
        linkage_matrix = sch.linkage(df_features, method=params_current['linkage'], metric=params_current['metrics'])
        labels = sch.fcluster(linkage_matrix, params_current['n_clusters'], criterion='maxclust')
        df['Cluster'] = labels

    elif algorithm == 'Birch Clustering':
        model = Birch(n_clusters=params_current['n_clusters'], threshold=params_current['threshold'], branching_factor=params_current['branching_factor'])
        labels = model.fit_predict(df_features)
        df['Cluster'] = labels

    else:
        if algorithm == 'HDBSCAN':
            model = hdbscan.HDBSCAN(min_cluster_size=params_current['min_cluster_size'], min_samples=params_current['min_samples'])
            labels = model.fit_predict(df_features)
        elif algorithm == 'Spectral Clustering':
            model = SpectralClustering(n_clusters=params_current['n_clusters'], affinity=params_current['affinity'], random_state=42)
            labels = model.fit_predict(df_features)
        elif algorithm == 'Ward Clustering':
            model = AgglomerativeClustering(n_clusters=params_current['n_clusters'], linkage='ward')
            labels = model.fit_predict(df_features)
        elif algorithm == 'Gaussian Mixture Models (GMM)':
            model = GaussianMixture(
                n_components=params_current['n_components'],
                covariance_type=params_current['covariance_type'],
                reg_covar=params_current['reg_covar'],
                random_state=42
            )
            labels = model.fit_predict(df_features)
            df['Cluster'] = labels

    df_plot = df.copy()
    df_plot['Cluster'] = labels
    col.scatter_chart(df_plot, x=df.columns[0], y=df.columns[1], color='Cluster', height=400, use_container_width=True)

    metrics = {}
    if len(set(labels)) > 1:
        metrics['Silhouette Score'] = silhouette_score(df_features, labels)
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(df_features, labels)
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(df_features, labels)
    else:
        metrics['Silhouette Score'] = 'Not enough clusters'
        metrics['Davies-Bouldin Index'] = 'Not enough clusters'
        metrics['Calinski-Harabasz Index'] = 'Not enough clusters'

    if algorithm == 'HDBSCAN':
        metrics['Stability Score'] = model.cluster_persistence_.mean() if model.cluster_persistence_ is not None else None

    if algorithm == 'Gaussian Mixture Models (GMM)':
        metrics['AIC value'] = model.aic(df_features)
        metrics['BIC value'] = model.bic(df_features)

    return metrics, params_current



with left_col:
    st.subheader(f"Algo 1: {algo1}")
    metrics_algo1, params_algo1 = perform_clustering(algo1, params, left_col)
    if all(isinstance(v, (int, float)) for v in metrics_algo1.values()):
        st.markdown("### Clustering Metrics")
        for metric, value in metrics_algo1.items():
            st.metric(metric, f"{value:.2f}")
    else:
        st.write("Metrics not available for this algorithm.")
        
with right_col:
    st.subheader(f"Algo 2: {algo2}")
    metrics_algo2, params_algo2 = perform_clustering(algo2, params, right_col)
    if all(isinstance(v, (int, float)) for v in metrics_algo2.values()):
        st.markdown("### Clustering Metrics")
        for metric, value in metrics_algo2.items():
            st.metric(metric, f"{value:.2f}")
    else:
        right_col.write("Metrics not available for this algorithm.")
