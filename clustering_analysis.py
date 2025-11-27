import numpy as np
from sklearn.cluster import KMeans
from collections import Counter, defaultdict

def run_clustering(feature_store, n_clusters=10):
    from sklearn.cluster import KMeans
    import numpy as np
    song_ids = list(feature_store.keys())
    X = np.array([fs['features']['chroma'].mean(axis=1) for fs in feature_store.values()])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    labels = kmeans.labels_
    cluster_map = {song_id: int(label) for song_id, label in zip(song_ids, labels)}

    return kmeans, cluster_map
