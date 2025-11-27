import networkx as networkx_graph_library
from fingerprint_index import compute_song_fingerprint_from_features, find_top_similar_songs_from_index

def build_influence_graph(song_feature_dictionary, fingerprint_index_dictionary, number_of_links_per_song=5):
    """ this function onstructs a directed influence graph where each song is connected to its top similar songs.
    The edges are weighted by similarity scoree, it takes
        song_feature_dictionary which mapps each song path to its feature set and genre.
        fingerprint_index_dictionary (which maaps each song path to its precomputed fingerprint.
        number_of_links_per_song (which is the number of top similar neighbors to connect each node to.

        it returns a 
        networkx.DiGraph: Directed graph with songs as nodes and similarity edges.
    """
    print("\nbuilding influence graph based on fingerprint similarity")

    #init an empty directed graph where we'll add each song and its connections
    directed_similarity_graph = networkx_graph_library.DiGraph()
    for current_song_path in song_feature_dictionary:
        # computer the fingerprint of the current song from its chroma features
        current_song_chroma_features = song_feature_dictionary[current_song_path]['features']['chroma']
        current_song_fingerprint_vector = compute_song_fingerprint_from_features(current_song_chroma_features)
        #get a list of most similar songs using cosine similarity
        most_similar_songs = find_top_similar_songs_from_index(current_song_fingerprint_vector, fingerprint_index_dictionary)
        #filter to avoid linking to itself, and limit to top-k results
        top_similar_neighbors = [
            (matched_song_path, similarity_score)
            for matched_song_path, similarity_score in most_similar_songs
            if matched_song_path != current_song_path
        ][:number_of_links_per_song]
        #add edges to the graph from the current song to each of its top neighbors
        for similar_song_path, similarity_score in top_similar_neighbors:
            directed_similarity_graph.add_edge(
                current_song_path,
                similar_song_path,
                weight=similarity_score  #use similarty as edge weight
            )

    return directed_similarity_graph


def compute_pagerank(graph_of_songs_with_similarities):
    """ applies the PageRank algorithm on the influence graph to compute song centrality
    based on how influential a song is within the overall similarity structure.
    takes
        graph_of_songs_with_similarities (networkx.DiGraph): Directed similarity graph.
returns:
        dict: Dictionary mapping song path to its PageRank score.
    """
    print("computing PageRank centrality scores.")
    return networkx_graph_library.pagerank(
        graph_of_songs_with_similarities,
        weight='weight'  #use edge weight (similarity) in PageRank calculation
    )
