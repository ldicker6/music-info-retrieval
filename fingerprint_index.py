from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# This function takes a dictionary of lowlevel audio features for a song
# and computes a single high-dimensional fingerprint vector by combining
def compute_song_fingerprint_from_features(song_feature_dictionary):
    #Compute the mean chroma vector (12-bin pitch class energy)
    mean_chroma_vector = song_feature_dictionary["chroma"].mean(axis=1)
    #Compute the mean spectral contrast (difference between peaks and valleys in each sub-band)
    mean_contrast_vector = song_feature_dictionary["contrast"].mean(axis=1)
    #compute the mean spectral centroid (weighted average of the frequencies present)
    average_centroid_value = np.mean(song_feature_dictionary["centroid"])
    
    #compute the mean spectral bandwidth (spread of frequencies around the centroid)
    average_bandwidth_value = np.mean(song_feature_dictionary["bandwidth"])
    
    #compute the mean spectral rolloff 
    average_rolloff_value = np.mean(song_feature_dictionary["rolloff"])
    
    #grab tempo
    tempo_estimate = song_feature_dictionary["tempo"]
    
    #put all feature components into a single vector
    return np.concatenate([
        mean_chroma_vector, 
        mean_contrast_vector, 
        [average_centroid_value, average_bandwidth_value, average_rolloff_value, tempo_estimate]
    ])

def compute_cosine_similarity_between_fingerprints(fingerprint_vector_1, fingerprint_vector_2):
    return cosine_similarity(fingerprint_vector_1.reshape(1, -1),  fingerprint_vector_2.reshape(1, -1))[0][0]

#this is used later for retrieval, matching queries against stored songs
def build_song_fingerprint_index(feature_store_by_song_id):
    song_fingerprint_index = {}
    
    for current_song_id, feature_metadata in feature_store_by_song_id.items():
        #recompute the fingerprint based on extracted features 
        fingerprint_vector = compute_song_fingerprint_from_features(feature_metadata['features'])
        #store the result in the index with the song's full path as the key
        song_fingerprint_index[current_song_id] = fingerprint_vector
    return song_fingerprint_index

# this function returns a sorted list of song matches (above a similarity threshold) 
def find_top_similar_songs_from_index(query_fingerprint_vector, fingerprint_index, similarity_threshold=0.85):
    list_of_similar_songs = []
    for indexed_song_id, indexed_fingerprint_vector in fingerprint_index.items():
        similarity_score = compute_cosine_similarity_between_fingerprints(
            query_fingerprint_vector, 
            indexed_fingerprint_vector)
        # only consider matches above the threshold for quality control
        if similarity_score > similarity_threshold:
            list_of_similar_songs.append((indexed_song_id, similarity_score))
    
    # sort by similarity score from highest to lowest
    list_of_similar_songs.sort(key=lambda match: match[1], reverse=True)
    return list_of_similar_songs
