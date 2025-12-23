from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

def run_clustering(X):
    results = {}

    kmeans = KMeans(n_clusters=6, random_state=42)
    k_labels = kmeans.fit_predict(X)
    results["kmeans"] = {
        "labels": k_labels,
        "silhouette": silhouette_score(X, k_labels),
        "db_index": davies_bouldin_score(X, k_labels)
    }

    dbscan = DBSCAN(eps=0.02, min_samples=100)
    d_labels = dbscan.fit_predict(X)
    results["dbscan"] = {"labels": d_labels}

    hierarchical = AgglomerativeClustering(n_clusters=6)
    h_labels = hierarchical.fit_predict(X)
    results["hierarchical"] = {
        "labels": h_labels,
        "silhouette": silhouette_score(X, h_labels)
    }

    return results