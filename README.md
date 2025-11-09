# 🍷 Clustering Algorithm Dashboard

An interactive Streamlit dashboard that visualizes, compares, and explains the behavior of multiple **clustering algorithms** on dimensionally reduced wine datasets.

Built as part of an unsupervised learning project to help users **see how different algorithms form clusters**, how well they perform, and which settings work best—without needing to dig through raw code or metrics.

👉 **Live demo:** [Wine Clustering Dashboard](https://wineclustering-caitq9txjnh6mm9apbpfd6.streamlit.app/)

---

## 👥 Developed By

* **Kong Jia Le** (Leader)
* **Jerry Tay Kien Hui**
* **Chan Wei Xin**
* **Tong Chun Mun**

---

## 🌐 Overview

The system provides three main views:

1. **Clustering Dashboard** – Try different algorithms, tweak parameters, and see how clusters & metrics react in real time.
2. **Algorithm Comparison** – Place two algorithms side-by-side, compare their clusters and evaluation scores.
3. **Results & Conclusions** – Visual + textual summary of which setups performed best and why.

It’s designed as a **learning tool** and **result viewer** for anyone exploring clustering quality, dataset preprocessing choices, and evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, etc.).

---

## 🔍 Supported Algorithms

Across the pages, the app showcases:

* Hierarchical Clustering
* Ward Clustering
* Spectral Clustering
* HDBSCAN
* BIRCH
* Gaussian Mixture Models (GMM)

Each algorithm is applied on preprocessed datasets (`MinMax_ICA`, `Robust_ICA`, `Zscore_ICA`) to highlight how **scaling + dimensionality reduction** influence clustering behavior.

---

## 🖥️ Page Breakdown

### 1. Clustering Dashboard

Interactive playground to **experiment** with:

* Algorithm selection via sidebar
* Dataset selection (`MinMax`, `Robust`, `Zscore`)
* Tunable parameters:

  * e.g. `n_clusters`, linkage type, affinity, thresholds, min_samples, covariance type, etc.
* Visual outputs:

  * Cluster scatter plots
  * Dendrograms (for hierarchical/ward)
* Metrics panel:

  * **Silhouette Score**
  * **Davies-Bouldin Index**
  * **Calinski-Harabasz Index**
  * HDBSCAN stability
  * GMM AIC/BIC
* Distribution table:

  * Cluster label vs data point counts with progress-bar style visualization

This page helps users see **“how sensitive is this method?”** and **“do the clusters look meaningful?”**.

---

### 2. Algorithm Comparison

Side-by-side comparison of **two algorithms at a time**:

* Fixed, sensible parameter presets per algorithm
* Automatic clustering on chosen mapped datasets
* Scatter plots for both algorithms
* Metrics shown per side:

  * Silhouette, Davies-Bouldin, Calinski-Harabasz
  * HDBSCAN Stability (if applicable)
  * GMM AIC/BIC (if applicable)

Useful for:

* Quickly explaining trade-offs between methods
* Supporting your written findings with visual evidence

---

### 3. Results & Conclusions

A guided summary page that:

* Runs each algorithm with selected “best” configs
* Plots final clusters using Altair
* Shows key metrics in styled blocks
* Provides concise explanations, such as:

  * Which dataset variant performed best (e.g. MinMax ICA)
  * Which algorithm gave the most compact & well-separated clusters
  * When density-based vs model-based vs hierarchical methods are appropriate

This page acts as the **storytelling layer**: it ties numbers, visuals, and interpretation together for reports or presentations.

---

## 📦 Requirements

Make sure you have:

* Python **3.9+**
* The following libraries installed:

```txt
streamlit
pandas
matplotlib
scikit-learn
hdbscan
scipy
altair
```

Ensure the CSV files used in the app are available in the same directory:

* `MinMax_ICA.csv`
* `Robust_ICA.csv`
* `Zscore_ICA.csv`

---

## 🚀 How to Run

1. (Optional) Create & activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

*(or install the listed packages manually)*

3. Launch each module with Streamlit:

```bash
streamlit run Dashboard.py
# or
streamlit run Comparison.py
# or
streamlit run Conclusion.py
```

Open the provided local URL in your browser and start exploring.

---

## ✅ Purpose

This dashboard is intended as a **teaching and analysis companion**:

* To make clustering results transparent and intuitive
* To help compare algorithms beyond theory—using real metrics & visuals
* To support your unsupervised learning findings in a clean, interactive way

