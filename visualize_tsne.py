import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import seaborn as sns

sns.set(style="whitegrid")
def load_metadata():
    with open(os.path.join(os.path.dirname(__file__), "results", "agent_metadata.pkl"), "rb") as f:
        return pickle.load(f)
def main():
    print("run t-SNE visualization, to be saved")
    metadata = load_metadata()
    feature_store = metadata["feature_store"]
    cluster_map = metadata["cluster_map"]
    kmeans = metadata["kmeans_model"]
    #build matrix of chroma-mean vectors
    song_ids = []
    vectors = []
    genres = []
    for song_id, entry in feature_store.items():
        chroma = entry['features']['chroma']
        mean_vec = chroma.mean(axis=1)
        vectors.append(mean_vec)
        song_ids.append(song_id)
        genres.append(entry['genre'])
    X = np.array(vectors)
    #run tsne
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("hsv", len(set(genres)))
    genre_to_color = {genre: palette[i] for i, genre in enumerate(sorted(set(genres)))}
    for i, (x, y) in enumerate(X_tsne):
        plt.scatter(x, y, color=genre_to_color[genres[i]], alpha=0.6, edgecolor='k', s=35)
    #legend
    for genre, color in genre_to_color.items():
        plt.scatter([], [], color=color, label=genre)
    plt.legend(loc='upper right', title="Genres", fontsize=9)
    plt.title("Map of Songs by Genre (Chroma-Based)", fontsize=14)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    #save it
    output_path = os.path.join(os.path.dirname(__file__), "results", "tsne_map.png")
    plt.savefig(output_path)
    plt.show()
if __name__ == "__main__":
    main()
