import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan
import scipy.cluster.hierarchy as sch
import altair as alt

def perform_clustering(algorithm, df, params):
    params_current = params[algorithm]

    if algorithm == 'Hierarchical Clustering':
        linkage_matrix = sch.linkage(df, method=params_current['linkage'], metric=params_current['metrics'])
        labels = sch.fcluster(linkage_matrix, params_current['n_clusters'], criterion='maxclust')

    elif algorithm == 'HDBSCAN':
        model = hdbscan.HDBSCAN(min_cluster_size=params_current['min_cluster_size'], min_samples=params_current['min_samples'])
        labels = model.fit_predict(df)

    elif algorithm == 'Spectral Clustering':
        model = SpectralClustering(n_clusters=params_current['n_clusters'], affinity=params_current['affinity'], random_state=42)
        labels = model.fit_predict(df.iloc[:, :-1])

    elif algorithm == 'Ward Clustering':
        model = AgglomerativeClustering(n_clusters=params_current['n_clusters'], linkage=params_current['linkage'])
        labels = model.fit_predict(df.iloc[:, :-1])

    elif algorithm == 'Birch Clustering':
        model = Birch(n_clusters=params_current['n_clusters'], threshold=params_current['threshold'], branching_factor=params_current['branching_factor'])
        labels = model.fit_predict(df.iloc[:, :-1])

    elif algorithm == 'Gaussian Mixture Models (GMM)':
        model = GaussianMixture(n_components=params_current['n_components'], covariance_type=params_current['covariance_type'], reg_covar=params_current['reg_covar'], random_state=42)
        labels = model.fit_predict(df.iloc[:, :-1])

    col1, col2 = st.columns([0.9, 1])

    with col1:
        formatted_params = ', '.join([f"{key}: {value}" for key, value in params_current.items()])
        st.write(f"**Parameters:** [`{formatted_params}`]")

        if labels is not None:
            df['Cluster'] = labels

            chart = alt.Chart(df).mark_circle(size=100).encode(
                x=alt.X(df.columns[0]),
                y=alt.Y(df.columns[1]),
                color=alt.Color('Cluster:N', scale=alt.Scale(scheme='viridis'), legend=alt.Legend(title="Clusters"))
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

    with col2:
        silhouette_avg = silhouette_score(df.iloc[:, :-1], labels)
        davies_bouldin = davies_bouldin_score(df.iloc[:, :-1], labels)
        calinski_harabasz = calinski_harabasz_score(df.iloc[:, :-1], labels)

        st.markdown('#### Clustering Metrics Summary')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style='background-color: #003c00; padding: 10px; border-radius: 5px;'>
                <h5 style='color: #ccc;'>Common Metrics</h5>
                <ul>
                    <li><b>Silhouette Score:</b> <code>{:.2f}</code></li>
                    <li><b>Davies-Bouldin Index:</b> <code>{:.2f}</code></li>
                    <li><b>Calinski-Harabasz Index:</b> <code>{:.2f}</code></li>
                </ul>
            </div>
            """.format(silhouette_avg, davies_bouldin, calinski_harabasz), unsafe_allow_html=True)

        with col2:
            if algorithm == 'HDBSCAN':
                stability_score = model.cluster_persistence_.mean() if model.cluster_persistence_ is not None else 'N/A'
                st.markdown("""
                <div style='background-color: #b34e24; padding: 10px; border-radius: 5px;'>
                    <h5 style='color: #ccc;'>HDBSCAN Specific Metrics</h3>
                    <ul>
                        <li><b>Stability Score:</b> <code>{:.2f}</code></li>
                    </ul>
                </div>
                """.format(stability_score), unsafe_allow_html=True)

            elif algorithm == 'Gaussian Mixture Models (GMM)':
                aic = model.aic(df.iloc[:, :-1])
                bic = model.bic(df.iloc[:, :-1])
                st.markdown("""
                <div style='background-color: #482980; padding: 10px; border-radius: 5px;'>
                    <h5 style='color: #ccc;'>GMM Specific Metrics</h3>
                    <ul>
                        <li><b>AIC:</b> <code>{:.2f}</code></li>
                        <li><b>BIC:</b> <code>{:.2f}</code></li>
                    </ul>
                </div>
                """.format(aic, bic), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        if algorithm == 'Hierarchical Clustering':
            st.markdown("""
            The choice of **linkage method** and **distance metric** for hierarchical clustering significantly impacts the results. Based on the provided evaluation metrics, the **MinMax dataset** with **average linkage** and **cityblock distance**, and **3 clusters** appears to have the best overall performance. 

            - **Silhouette Score**: 0.56 (indicating reasonable clustering performance)
            - **Davies-Bouldin Index**: Lowest value
            - **Calinski-Harabasz Index**: Highest value

            These metrics indicate **well-separated and compact clusters** with the chosen algorithm performing reasonably well.
            """)

        elif algorithm == 'HDBSCAN':
            st.markdown("""
            The **Z Score dataset** with **min_cluster_size=20** and **min_samples=5** achieved the best overall performance. It achieves higher **Silhouette Score** and **Stability Score** compared to other datasets, indicating well-separated and stable clusters.

            - **Silhouette Score**: 0.41 (though not very high, it shows that the algorithm is still appropriate for this dataset)
            - **Davies-Bouldin Index**: Not Suitable for Density-based Clustering Algorithms
            - **Calinski-Harabasz Index**: Not Suitable for Density-based Clustering Algorithms
            - **Stability Score**: Higher than other configurations

            """)
            
        elif algorithm == 'Spectral Clustering':
            st.markdown("""
            Both the **MinMax ICA** and **Z Score ICA** datasets deliver strong clustering outcomes, with **MinMax ICA** slightly outperforming **Z Score ICA**. The **Robust ICA dataset** shows the weakest clustering performance.

            - **Silhouette Score**: 0.57 (on the MinMax dataset, indicating good performance)
            - **Calinski-Harabasz Index**: High
            - **Davies-Bouldin Index**: Low (indicating compact clusters)

            The **MinMax ICA** dataset shows strong **cohesion** and **separation**, resulting in well-defined clusters and overall good clustering performance.
            """)

        elif algorithm == 'Ward Clustering':
            st.markdown("""
            Similar to Spectral Clustering, **MinMax ICA** and **Z Score ICA** exhibit strong clustering outcomes, with **MinMax ICA** slightly outperforming. **Robust ICA** shows weaker performance.

            - **Silhouette Score**: 0.56 (on the MinMax ICA dataset)
            - **Calinski-Harabasz Index**: High
            - **Davies-Bouldin Index**: Low

            The **MinMax ICA** dataset has **well-defined clusters**, and the algorithm proves quite suitable for this dataset.
            """)

        elif algorithm == 'Birch Clustering':
            st.markdown("""
            After tuning the clustering parameters, the **MinMax ICA dataset** with **number of clusters = 3**, **threshold = 0.01**, and **branching factor = 20** achieved the best performance.

            - **Silhouette Score**: 0.56
            - **Calinski-Harabasz Index**: 306.10
            - **Davies-Bouldin Index**: 0.58
            
            BIRCH proves to be particularly well-suited for the **MinMax ICA dataset**, efficiently handling its structure and variability to achieve superior clustering results.
            """)

        elif algorithm == 'Gaussian Mixture Models (GMM)':
            st.markdown("""
            The **MinMax ICA dataset** provides the best overall clustering performance among the GMM configurations.

            - **Silhouette Score**: 0.56 (indicating strong cohesion and separation)
            - **Calinski-Harabasz Index**: High (well-defined clusters)
            - **Davies-Bouldin Index**: Low (compact clusters)
            - **AIC and BIC**: Lowest values, confirming optimal balance between model fit and complexity

            These results collectively demonstrate that the **MinMax ICA** dataset delivers the most effective clustering configuration in this analysis.
            """)

st.set_page_config(page_title="Clustering Result", layout="wide", page_icon="🍷")

# Load needed datasets
minmax_ica = pd.read_csv('MinMax_ICA.csv')
zscore_ica = pd.read_csv('Zscore_ICA.csv')

params = {
    'Hierarchical Clustering': {'n_clusters': 3, 'linkage': 'average', 'metrics': 'cityblock'},
    'HDBSCAN': {'min_cluster_size': 20, 'min_samples': 5},
    'Spectral Clustering': {'n_clusters': 3, 'affinity': 'rbf', 'n_neighbors': 5},
    'Ward Clustering': {'n_clusters': 3, 'linkage': 'average'},
    'Birch Clustering': {'n_clusters': 3, 'threshold': 0.05, 'branching_factor': 20},
    'Gaussian Mixture Models (GMM)': {'n_components': 3, 'covariance_type': 'diag', 'reg_covar': 0.00000001}
}

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .section-header { font-size: 1.75rem; font-weight: bold; margin-top: 2rem; }
    hr { border: none; height: 1px; background-color: #ccc; margin: 1.5rem 0; }
    </style>
    """, 
    unsafe_allow_html=True
)

st.markdown("# Clustering Algorithm Results Overview")
st.markdown("Below is a summary of clustering results using various algorithms. Each algorithm was applied to dimensionally reduced data.")
st.markdown('<hr>', unsafe_allow_html=True)

# Hierarchical Clustering
st.markdown('<div class="section-header">Hierarchical Clustering</div>', unsafe_allow_html=True)
perform_clustering('Hierarchical Clustering', minmax_ica, params)
st.markdown('<hr>', unsafe_allow_html=True)

# HDBSCAN
st.markdown('<div class="section-header">HDBSCAN</div>', unsafe_allow_html=True)
perform_clustering('HDBSCAN', zscore_ica, params)
st.markdown('<hr>', unsafe_allow_html=True)

# Spectral Clustering
st.markdown('<div class="section-header">Spectral Clustering</div>', unsafe_allow_html=True)
perform_clustering('Spectral Clustering', minmax_ica, params)
st.markdown('<hr>', unsafe_allow_html=True)

# Ward Clustering
st.markdown('<div class="section-header">Ward Clustering</div>', unsafe_allow_html=True)
perform_clustering('Ward Clustering', minmax_ica, params)
st.markdown('<hr>', unsafe_allow_html=True)

# Birch Clustering
st.markdown('<div class="section-header">Birch Clustering</div>', unsafe_allow_html=True)
perform_clustering('Birch Clustering', minmax_ica, params)
st.markdown('<hr>', unsafe_allow_html=True)

# Gaussian Mixture Models (GMM)
st.markdown('<div class="section-header">Gaussian Mixture Models (GMM)</div>', unsafe_allow_html=True)
perform_clustering('Gaussian Mixture Models (GMM)', minmax_ica, params)

st.markdown('<hr>', unsafe_allow_html=True)
