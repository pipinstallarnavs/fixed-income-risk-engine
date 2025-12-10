"""K-means clustering of issuers into spread-behaviour peer groups."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_issuers(issuers, n_clusters=6, seed=3):
    features = np.array([
        [iss["spread_level"] * 1e4, iss["spread_vol"] * 1e4, iss["spread_beta"]]
        for iss in issuers
    ])
    X = StandardScaler().fit_transform(features)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    labels = km.fit_predict(X)

    for iss, label in zip(issuers, labels):
        iss["cluster"] = int(label)

    return labels, km
