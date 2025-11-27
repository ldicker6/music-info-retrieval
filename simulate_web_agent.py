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
#format and save report
report_lines = []
report_lines.append("=" * 60)
report_lines.append(f"🎧  Query: {Path(held_out_song).name}")
report_lines.append(f"   • True genre: {held_out_genre}")
report_lines.append(f"   • Predicted genre: {predicted_genre}\n")

report_lines.append(f"🧬  Cluster ID: {cluster_id}")
report_lines.append("📊  Cluster composition (top genres):")
for genre, count in archetype:
    report_lines.append(f"   • {genre:<10} {count} songs")

report_lines.append("\n🔎  Top-5 similar tracks (by cosine similarity):")
for rank, (sid, score) in enumerate(cluster_matches, start=1):
    genre = feature_store[sid]["genre"]
    report_lines.append(f"   {rank}) {Path(sid).name:<30}  genre: {genre:<10}  similarity: {score:.4f}")

report_lines.append("=" * 60)
report_path = RESULT_DIR / "query_result_report_example.txt"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))

print(f"Simulation complete. report saved to {report_path}")