import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import hdbscan
import scipy.cluster.hierarchy as sch
import altair as alt

st.set_page_config(page_title="Clustering Dashboard", layout="wide", page_icon="🍷")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

# Load datasets
minmax_ica = pd.read_csv('MinMax_ICA.csv')
robust_ica = pd.read_csv('Robust_ICA.csv')
zscore_ica = pd.read_csv('Zscore_ICA.csv')
datasets = {
    'MinMax': minmax_ica,
    'Robust': robust_ica,
    'Zscore': zscore_ica
}

st.markdown("# Clustering Dashboard")

algorithm = st.sidebar.selectbox(
    'Select Clustering Algorithm',
    ['Hierarchical Clustering',
     'HDBSCAN',
     'Spectral Clustering',
     'Ward Clustering',
     'Birch Clustering',
     'Gaussian Mixture Models (GMM)']
)

selected_dataset = st.sidebar.selectbox('Select Dataset', ['MinMax', 'Robust', 'Zscore'])
df = datasets[selected_dataset]

params = {}
col = st.columns((1.5, 4.5, 2), gap='medium')

# donut chart to display silhouette score
def make_donut(silhouette_score, input_text):
    silhouette_score = max(min(silhouette_score, 1), 0)

    chart_color = ['#eee','#29b5e8'] 
    
    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [1 - silhouette_score, silhouette_score]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [1, 0]
    })
    
    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)
    
    text = plot.mark_text(
        align='center',
        color=chart_color[0],
        font="Arial",
        fontSize=28,
        fontWeight=500
    ).encode(text=alt.value(f'{silhouette_score:.2f}'))
    
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color=alt.Color("Topic:N",
                        scale=alt.Scale(
                            domain=[input_text, ''],
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)
    
    return plot_bg + plot + text

with col[1]:
    if algorithm == 'Hierarchical Clustering':
        with st.sidebar:
            params['n_clusters'] = st.slider('Number of Clusters', min_value=2, max_value=10, value=3)
            params['linkage'] = st.selectbox('Linkage', ['single', 'complete', 'average'], index=2)
            params['metrics'] = st.selectbox('Metrics', ['euclidean', 'cityblock', 'cosine'], index=1)
            plot_selection = st.radio('Select Plot', ['Cluster Plot', 'Dendrogram'])

        linkage_matrix = sch.linkage(df, method=params['linkage'], metric=params['metrics'])
        labels = sch.fcluster(linkage_matrix, params['n_clusters'], criterion='maxclust')
        df['Cluster'] = labels

        if plot_selection == 'Cluster Plot':
            st.markdown('#### Cluster Plot')
            df_plot = df.copy()
            df_plot['Cluster'] = labels
            st.scatter_chart(df_plot, x=df.columns[0], y=df.columns[1], color='Cluster', height=500, use_container_width=True)
        else:
            st.markdown('#### Dendrogram')
            fig_dendrogram = plt.figure(figsize=(15, 10))
            sch.dendrogram(linkage_matrix)
            plt.ylabel('Distance', fontsize=18)
            st.pyplot(fig_dendrogram)

    elif algorithm == 'Birch Clustering':
        with st.sidebar:
            params['n_clusters'] = st.slider('Number of Clusters', min_value=2, max_value=10, value=3)
            params['threshold'] = st.slider('Threshold', min_value=0.01, max_value=0.2, value=0.05)
            params['branching_factor'] = st.slider('Branching Factor', min_value=10, max_value=200, value=20)
            plot_selection = st.radio('Select Plot for Birch Clustering', ['Cluster Plot', 'CF Tree Plot'])
            
        model = Birch(n_clusters=params['n_clusters'], threshold=params['threshold'], branching_factor=params['branching_factor'])
        labels = model.fit_predict(df)
        df['Cluster'] = labels

        def plot_cf_tree(birch_model, X, dataset_name):
            try:
                subcluster_centers = birch_model.subcluster_centers_
            except AttributeError:
                st.write(f"No sub-clusters found in {dataset_name}.")
                return

            if subcluster_centers.size == 0:
                st.write(f"No sub-clusters found in {dataset_name}.")
                return

            plt.figure(figsize=(10, 4))
            plt.scatter(X[:, 0], X[:, 1], c='blue', label='Data Points', alpha=0.6)
            plt.scatter(subcluster_centers[:, 0], subcluster_centers[:, 1], c='red', marker='x', s=100, label='Sub-cluster Centers')
            plt.title(f'{dataset_name} - BIRCH CF Tree Inspection')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            st.pyplot(plt.gcf()) 


        if plot_selection == 'CF Tree Plot':
            st.markdown('#### CF Tree Plot')
            plot_cf_tree(model, df.iloc[:, :-1].values, selected_dataset)
        else:
            st.markdown('#### Cluster Plot')
            df_plot = df.copy()
            df_plot['Cluster'] = labels
            st.scatter_chart(df_plot, x=df.columns[0], y=df.columns[1], color='Cluster', height=500, use_container_width=True)


    elif algorithm == 'HDBSCAN':
        params['min_cluster_size'] = st.sidebar.slider('Min Cluster Size', min_value=5, max_value=50, value=20)
        params['min_samples'] = st.sidebar.slider('Min Samples', min_value=1, max_value=50, value=5)
        model = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'], min_samples=params['min_samples'])
        labels = model.fit_predict(df)
        df['Cluster'] = labels
        st.markdown('#### Cluster Plot')
        df_plot = df.copy()
        df_plot['Cluster'] = labels
        st.scatter_chart(df_plot, x=df.columns[0], y=df.columns[1], color='Cluster', height=500, use_container_width=True)

    elif algorithm == 'Spectral Clustering':
        params['n_clusters'] = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=3)
        params['affinity'] = st.sidebar.selectbox('Affinity', ['nearest_neighbors', 'rbf'])
        params['n_neighbors'] = st.sidebar.slider('Number of neighbors', min_value=2, max_value=20, value=5)
        model = SpectralClustering(n_clusters=params['n_clusters'], affinity=params['affinity'], n_neighbors=params['n_neighbors'], random_state=42)
        labels = model.fit_predict(df)
        df['Cluster'] = labels
        
        st.markdown('#### Cluster Plot')
        df_plot = df.copy()
        df_plot['Cluster'] = labels
        st.scatter_chart(df_plot, x=df.columns[0], y=df.columns[1], color='Cluster', height=500, use_container_width=True)
        
        
    elif algorithm == 'Ward Clustering':
        params['n_clusters'] = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=3)
        params['linkage'] = st.sidebar.selectbox('Linkage', ['single', 'complete', 'average', 'ward'], index=2)
        plot_selection = st.sidebar.radio('Select Plot for Ward Clustering', ['Cluster Plot', 'Dendrogram'])
        model = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params['linkage'])
        linkage_matrix = sch.linkage(df, method=params['linkage'])
        labels = sch.fcluster(linkage_matrix, params['n_clusters'], criterion='maxclust')
        df['Cluster'] = labels
        
        if plot_selection == 'Dendrogram':
            st.markdown('#### Dendrogram')
            fig_dendrogram = plt.figure(figsize=(15, 10))
            sch.dendrogram(linkage_matrix)
            plt.ylabel('Distance', fontsize=18)
            st.pyplot(fig_dendrogram)
        
        else:
            st.markdown('#### Cluster Plot')
            df_plot = df.copy()
            df_plot['Cluster'] = labels
            st.scatter_chart(df_plot, x=df.columns[0], y=df.columns[1], color='Cluster', height=500, use_container_width=True)


    elif algorithm == 'Gaussian Mixture Models (GMM)':
        params['n_components'] = st.sidebar.slider('Number of Components', min_value=2, max_value=10, value=3)
        params['covariance_type'] = st.sidebar.selectbox('Covariance Type', ['full', 'tied', 'diag', 'spherical'], index=2)
        reg_covar_options = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
        params['reg_covar'] = st.sidebar.select_slider('Regularization Covariance', options=reg_covar_options, value=0.00000001)
        model = GaussianMixture(n_components=params['n_components'], covariance_type=params['covariance_type'], reg_covar=params['reg_covar'], random_state=42)
        labels = model.fit_predict(df)
        df['Cluster'] = labels
        df_features_only = df.drop(columns=['Cluster'])
        aic_value = model.aic(df_features_only)
        bic_value = model.bic(df_features_only)

        st.markdown('#### Cluster Plot')
        df_plot = df.copy()
        df_plot['Cluster'] = labels
        st.scatter_chart(df_plot, x=df.columns[0], y=df.columns[1], color='Cluster', height=500, use_container_width=True)

if len(set(df['Cluster'])) > 1:  
    silhouette_score = silhouette_score(df.iloc[:, :-1], df['Cluster'])
    davies_bouldin = davies_bouldin_score(df.iloc[:, :-1], df['Cluster'])
    calinski_harabasz = calinski_harabasz_score(df.iloc[:, :-1], df['Cluster'])
    
else:
    silhouette_score = None
    davies_bouldin = None
    calinski_harabasz = None


with col[0]:
    st.markdown('#### Performance')
    
    if silhouette_score is not None:
        st.markdown("Silhouette Score")
        st.altair_chart(make_donut(silhouette_score, input_text="Silhouette"), use_container_width=True)
    else:
        st.write("Silhouette Score: -")

    if davies_bouldin is not None:
        st.metric(label="Davies-Bouldin Index", value=f"{davies_bouldin:.2f}")
    else:
        st.write("Davies-Bouldin Index: -")

    if calinski_harabasz is not None:
        st.metric(label="Calinski-Harabasz Index", value=f"{calinski_harabasz:.2f}")
    else:
        st.write("Calinski-Harabasz Index: -")

    if algorithm == 'HDBSCAN':
        stability_score = model.cluster_persistence_.mean()
        if stability_score is not None:
            st.metric(label="Stability Score", value=f"{stability_score:.3f}")
        else:
            st.write("Stability Score: -")

    if algorithm == 'Gaussian Mixture Models (GMM)':
        st.metric(label="AIC Score", value=f"{aic_value:.3f}")
        st.metric(label="BIC Score", value=f"{bic_value:.3f}")




with col[2]:
    label_counts = df['Cluster'].value_counts().reset_index()
    label_counts.columns = ['Cluster Label', 'Count']

    max_count = max(label_counts['Count'])
    
    st.markdown("#### Datapoint Distribution")
    st.dataframe(
        label_counts,
        hide_index=True,
        column_order=("Cluster Label", "Count"),
        column_config={
            "Cluster Label": st.column_config.TextColumn(
                "Cluster Label",
            ),
            "Count": st.column_config.ProgressColumn(
                "Data Points Count",
                format="%d",
                min_value=0,
                max_value=max_count
            )
        },
        use_container_width=True
    )


