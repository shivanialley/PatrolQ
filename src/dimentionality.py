from sklearn.decomposition import PCA
import json
import os

def apply_pca(X):
    os.makedirs("artifacts", exist_ok=True)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    with open("artifacts/pca_variance.json", "w") as f:
        json.dump(pca.explained_variance_ratio_.tolist(), f)

    return X_pca, pca

