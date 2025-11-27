# simulate_web_agent.py

import os
import random
import pickle
import numpy as np
from collections import Counter
from pathlib import Path
from audio_processing import extract_features
from fingerprint_index import compute_song_fingerprint_from_features, find_top_similar_songs_from_index

PROJECT_DIR = Path(__file__).parent
RESULT_DIR = PROJECT_DIR / "results" / "simulation_output"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_DIR / "results" / "agent_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

feature_store = metadata["feature_store"]
index = metadata["index"]
pagerank = metadata["pagerank"]
cluster_map = metadata["cluster_map"]
kmeans = metadata["kmeans_model"]
#pick a random song
held_out_song = random.choice(list(feature_store.keys()))
held_out_features = feature_store[held_out_song]["features"]
held_out_genre = feature_store[held_out_song]["genre"]
held_out_fp = compute_song_fingerprint_from_features(held_out_features["chroma"])
#predict genre
matches = find_top_similar_songs_from_index(held_out_fp, index)[:5]
top_genres = [feature_store[sid]["genre"] for sid, _ in matches]
predicted_genre = Counter(top_genres).most_common(1)[0][0]
#assign to cluster manually -- because of bug
vector = held_out_features["chroma"].mean(axis=1).reshape(1, -1)
cluster_id = int(np.argmin(np.linalg.norm(kmeans.cluster_centers_ - vector, axis=1)))
# cluster archetype
cluster_genres = [
    feature_store[sid]["genre"] for sid, cid in cluster_map.items() if cid == cluster_id
]
archetype = Counter(cluster_genres).most_common(3)
# find more songs in the same cluster
cluster_matches = [
    (sid, sim) for sid, sim in find_top_similar_songs_from_index(held_out_fp, index)
    if cluster_map.get(sid) == cluster_id and sid != held_out_song
][:5]
# ✏️ Write report
report_path = RESULT_DIR / "query_result_report_example.txt"
with open(report_path, "w") as f:
    f.write(f"Query Song: {held_out_song}\n")
    f.write(f"Actual Genre: {held_out_genre}\n")
    f.write(f"predicted Genre: {predicted_genre}\n\n")
    f.write(f"cluster ID: {cluster_id}\n")
    f.write(f"cluster Archetype (Top Genres):\n")
    for g, c in archetype:
        f.write(f"   - {g}: {c} songs\n")
    f.write("\nmore Songs from Same Cluster (Top-5 by Similarity):\n")
    for sid, sim in cluster_matches:
        genre = feature_store[sid]["genre"]
        f.write(f"   - {sid} — Genre: {genre} — Similarity: {sim:.4f}\n")

print(f"simulation complete and saved to {report_path}")
