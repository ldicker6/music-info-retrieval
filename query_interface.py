import os
import sys
import pickle
import numpy as np
from collections import Counter

# Custom project imports for feature extraction and audio fingerprinting
from audio_processing import extract_features
from fingerprint_index import compute_song_fingerprint_from_features, find_top_similar_songs_from_index

def run_query_agent_with_friendly_output(query_song_file_path):
    """this function is the main 'user agent' of the system, it can loads all of the necessary
    model data and metadata that was previously saved after training,  extracts features
    from a user-provided song (via command line), and then it will coomputes its fingerprint, finds the top-5 most similar songs using cosine similarity predict the genre, identify the song's cluster
    prrint the genre distribution (archetype) of that cluster and recommends 5 similar songs from that same cluster
    """
    #laod the saved training data and clustering models from a single pickled metadata file
    path_to_agent_metadata_pickle = os.path.join(
        os.path.dirname(__file__), "results", "agent_metadata.pkl")
    with open(path_to_agent_metadata_pickle, "rb") as metadata_file:
        all_cached_metadata = pickle.load(metadata_file)
    #unpack everything we need from the stored bundle
    feature_dictionary_for_all_songs = all_cached_metadata["feature_store"]
    training_fingerprint_index = all_cached_metadata["index"]
    pagerank_scores_for_graph = all_cached_metadata["pagerank"]
    cluster_id_assignments_for_all_songs = all_cached_metadata["cluster_map"]
    trained_kmeans_model_object = all_cached_metadata["kmeans_model"]
    print(f"\nquerying with: {query_song_file_path}")
    #extract audio features from the song the user has chosen
    extracted_feature_dictionary_for_query_song = extract_features(query_song_file_path)
    #compute the fingerprint from extracted features
    fingerprint_vector_for_query_song = compute_song_fingerprint_from_features(
        extracted_feature_dictionary_for_query_song
    )
    #Search for the top-5 most similar songs in the training set index
    top_5_matches_by_similarity = find_top_similar_songs_from_index(
        fingerprint_vector_for_query_song, training_fingerprint_index
    )[:5]
    print(f"\ntop-5 similar songs:")
    for matched_song_id, similarity_score in top_5_matches_by_similarity:
        genre_of_matched_song = feature_dictionary_for_all_songs[matched_song_id]["genre"]
        print(f"  {matched_song_id} — Genre: {genre_of_matched_song} — Similarity: {similarity_score:.4f}")

    #determine the predicted genre based on top matches
    genres_of_top_matches = [
        feature_dictionary_for_all_songs[match[0]]["genre"]
        for match in top_5_matches_by_similarity
    ]
    if genres_of_top_matches:
        most_common_predicted_genre = Counter(genres_of_top_matches).most_common(1)[0][0]
        print(f"\npredicted Genre: {most_common_predicted_genre}")
    else:
        print("\npredicted Genre: Unknown")

    #determine which cluster this query song would belong to
    #convert chroma matrix to a  vector and compare to centroids
    average_chroma_vector = extracted_feature_dictionary_for_query_song["chroma"].mean(axis=1).reshape(1, -1)
    distances_to_each_centroid = np.linalg.norm(
        trained_kmeans_model_object.cluster_centers_ - average_chroma_vector,
        axis=1
    )
    assigned_cluster_id_for_query = int(np.argmin(distances_to_each_centroid))

    print(f"\ncluster ID: {assigned_cluster_id_for_query}")

    #nromalize all cluster keys to avoid mismatch due to path formatting
    normalized_cluster_mapping = {
        os.path.normpath(str(song_path)): cluster_id
        for song_path, cluster_id in cluster_id_assignments_for_all_songs.items()
    }
    #analyze genre distribution within the query's cluster
    genre_distribution_in_same_cluster = Counter(
        feature_dictionary_for_all_songs[song]["genre"]
        for song, cluster_id in normalized_cluster_mapping.items()
        if cluster_id == assigned_cluster_id_for_query
    )

    print(f"📊 Cluster Archetype (Top Genres):")
    if genre_distribution_in_same_cluster:
        for genre, number_of_songs in genre_distribution_in_same_cluster.most_common(3):
            print(f"   {genre}: {number_of_songs} songs")
    else:
        print("   [No archetype data available]")

    #recommend other songs in the same cluster that are also similar
    print(f"\n🎧 More Songs from Same Cluster (sorted by similarity):")
    similar_songs_from_same_cluster = []
    for candidate_song_id, similarity_score in find_top_similar_songs_from_index(
        fingerprint_vector_for_query_song, training_fingerprint_index
    ):
        #normalize the song path to match keys in the cluster map
        normalized_candidate_song_id = os.path.normpath(candidate_song_id)
        if ( normalized_cluster_mapping.get(normalized_candidate_song_id) == assigned_cluster_id_for_query and candidate_song_id != query_song_file_path ):
            similar_songs_from_same_cluster.append((candidate_song_id, similarity_score))
        if len(similar_songs_from_same_cluster) >= 5:
            break

    if similar_songs_from_same_cluster:
        for similar_song_id, similarity_score in similar_songs_from_same_cluster:
            genre_of_similar_song = feature_dictionary_for_all_songs[similar_song_id]["genre"]
            print(f"  {similar_song_id} — Genre: {genre_of_similar_song} — Similarity: {similarity_score:.4f}")
    else:
        print("no similar songs found in the same cluster.")


def main():
    """  this is the entry point of the command-line interface.
    it expects  one argument: the path to the audio file to query,
    """
    if len(sys.argv) != 2:
        print("Usage: python query_interface.py path/to/query_song.au")
        return
    path_to_song_to_query = sys.argv[1]
    run_query_agent_with_friendly_output(path_to_song_to_query)

if __name__ == "__main__":
    main()
