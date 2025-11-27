import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from audio_processing import extract_features
from fingerprint_index import compute_song_fingerprint_from_features, find_top_similar_songs_from_index
from clustering_analysis import run_clustering
from influence_graph import build_influence_graph, compute_pagerank

#ensure a folder exists to save all the output results 
output_directory_path = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_directory_path, exist_ok=True)

def extract_all_features_and_fingerprints_from_dataset():
    """this function loads all songs in the dataset directory, extracts features for each one
    using the extract_features() function, and computes their fingerprint vectors.
    Returns:
        song_feature_dictionary, a dictionary mapping each song path to its features and genre.
        song_fingerprint_index, a dictionary mapping each song path to its fingerprint vector.
    """
    song_feature_dictionary = {}
    dataset_path = os.path.join(os.path.dirname(__file__), "data", "songs")
    all_audio_file_paths = glob.glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)
    for path_to_audio_file in all_audio_file_paths:
        if not path_to_audio_file.endswith((".au", ".wav", ".mp3")):
            continue  # skip files that aren't audio
        full_song_id_path = os.path.normpath(path_to_audio_file)
        genre_folder_name = os.path.basename(os.path.dirname(path_to_audio_file))
        #extract and store all features + the genre label
        song_feature_dictionary[full_song_id_path] = {
            'features': extract_features(path_to_audio_file),
            'genre': genre_folder_name
        }

    #create a dictionary of song fingerprints
    song_fingerprint_index = {
        song_path: compute_song_fingerprint_from_features(data['features'])
        for song_path, data in song_feature_dictionary.items()
    }
    return song_feature_dictionary, song_fingerprint_index


def main():
    print("\nExtracting features and fingerprints from dataset")
    song_feature_dictionary, song_fingerprint_index_all = extract_all_features_and_fingerprints_from_dataset()
    print(f"Total number of songs successfully processed: {len(song_feature_dictionary)}")

    #Split the dataset into training and testing portions
    all_song_paths = list(song_feature_dictionary.keys())
    random.seed(42)  # for reproducibility
    random.shuffle(all_song_paths)
    split_index = int(0.8 * len(all_song_paths))
    training_song_paths = all_song_paths[:split_index]
    testing_song_paths = all_song_paths[split_index:]
    print(f"Training set: {len(training_song_paths)} songs ,testing set: {len(testing_song_paths)} songs")
    fingerprint_index_for_training = {
        song_path: song_fingerprint_index_all[song_path] for song_path in training_song_paths
    }

    #rrun top-k genre retrieval evaluation
    correct_predictions_by_k = {1: 0, 3: 0, 5: 0}
    actual_genres_list = []
    predicted_genres_list = []

    print("\nPerforming Top-k genre match evaluation.")

    for query_song_path in testing_song_paths:
        query_fingerprint = compute_song_fingerprint_from_features(song_feature_dictionary[query_song_path]['features'])
        true_genre_label = song_feature_dictionary[query_song_path]['genre']
        similarity_ranked_matches = find_top_similar_songs_from_index(query_fingerprint, fingerprint_index_for_training)

        #filter out the  same song and duplicates
        top_5_distinct_matches = []
        for matched_song_path, similarity_score in similarity_ranked_matches:
            if matched_song_path != query_song_path and not np.allclose(fingerprint_index_for_training[matched_song_path], query_fingerprint, atol=1e-4):
                top_5_distinct_matches.append((matched_song_path, similarity_score))
            if len(top_5_distinct_matches) == 5:
                break
        # et genres of the top-5 matches
        top_5_genre_predictions = [song_feature_dictionary[path]['genre'] for path, _ in top_5_distinct_matches]
        for k in [1, 3, 5]:
            top_k_predictions = top_5_genre_predictions[:k]
            correct_predictions_by_k[k] += top_k_predictions.count(true_genre_label) / k
        if top_5_genre_predictions:
            most_common_genre = Counter(top_5_genre_predictions).most_common(1)[0][0]
        else:
            most_common_genre = "None"

        actual_genres_list.append(true_genre_label)
        predicted_genres_list.append(most_common_genre)

    # prrint accuracy metrics
    total_evaluation_samples = len(testing_song_paths)
    print(f"\nTop-k Genre Retrieval Accuracy:")
    for k in [1, 3, 5]:
        average_precision = correct_predictions_by_k[k] / total_evaluation_samples
        print(f"  Precision@{k}: {average_precision:.2f}")
    # create and save confusion matrix
    genre_label_set = sorted(set(actual_genres_list + predicted_genres_list))
    genre_confusion_matrix = confusion_matrix(actual_genres_list, predicted_genres_list, labels=genre_label_set)
    visualizer = ConfusionMatrixDisplay(confusion_matrix=genre_confusion_matrix, display_labels=genre_label_set)
    visualizer.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix: Genre Retrieval")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory_path, "confusion_matrix.png"))
    plt.close()
    #run clustering and build cluster metadata
    trained_kmeans_model, song_to_cluster_map = run_clustering(song_feature_dictionary, n_clusters=10)
    #build influence graph and run PageRank
    influence_graph = build_influence_graph(song_feature_dictionary, song_fingerprint_index_all)
    pagerank_scores = compute_pagerank(influence_graph)
    # mnromalizee paths in the cluster map
    normalized_song_to_cluster_map = {
        os.path.normpath(song_path): cluster_id for song_path, cluster_id in song_to_cluster_map.items()
    }

    #save everything requiredfor agent
    agent_metadata_bundle = {
        "feature_store": song_feature_dictionary,
        "index": fingerprint_index_for_training,
        "pagerank": pagerank_scores,
        "cluster_map": normalized_song_to_cluster_map,
        "kmeans_model": trained_kmeans_model
    }
    metadata_path = os.path.join(output_directory_path, "agent_metadata.pkl")
    with open(metadata_path, "wb") as metadata_file:
        pickle.dump(agent_metadata_bundle, metadata_file)

if __name__ == "__main__":
    main()
