from collections import Counter
import itertools
import pandas as pd
from typing import List
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
import numpy as np
from clustering_stage_1 import NUM_OF_LABELS, KMeans1Result, read_kmeans1_results


def clustering_for_all_k(features: np.ndarray) -> List[KMeans]:
    kmeans_list = []
    max_clusters = min(len(features), NUM_OF_LABELS)
    for clusters in range(2, max_clusters):
        kmeans = KMeans(n_clusters=clusters, n_init="auto").fit(features)
        kmeans_list.append(kmeans)

    return kmeans_list


def get_statistics(kmeans_list: List[KMeans], features_array: np.ndarray) -> pd.DataFrame:
    inertias = []
    k_values = []
    silhouette_scores = []
    biggest_cluster_sizes = []
    samples_counts = [len(features_array)] * len(kmeans_list)
    for kmeans in kmeans_list:
        k_values.append(kmeans.n_clusters)
        inertias.append(kmeans.inertia_)
        silhouette_score_value = silhouette_score(features_array, kmeans.labels_, metric="euclidean")
        silhouette_scores.append(silhouette_score_value)
        _, biggest_cluster_size = Counter(kmeans.labels_).most_common(1)[0]
        biggest_cluster_sizes.append(biggest_cluster_size)

    return pd.DataFrame(
        {
            "k": k_values,
            "inertia": inertias,
            "silhouette_score": silhouette_scores,
            "biggest_cluster_size": biggest_cluster_sizes,
            "samples_count": samples_counts,
        }
    )


def get_kMeans1Results_iterator_grouped_by_predicted_label(results: List[KMeans1Result]):
    keyfunc = lambda x: x.predicted_label
    result_tuples = sorted(results, key=keyfunc)
    it = itertools.groupby(result_tuples, key=keyfunc)
    return it


if __name__ == "__main__":
    print("Loading kmeans1 results..")
    kMeans1Results = read_kmeans1_results("clustering_1/kmeans1_results.d")
    print("Data loaded")

    print("Running kmeans2 for all k")
    iterator = get_kMeans1Results_iterator_grouped_by_predicted_label(kMeans1Results)
    for predicted_label, kmeans1ResultIterator in iterator:
        print(f"Runnig kmeans2 stage: {predicted_label}/{NUM_OF_LABELS}")
        kmeans1ResultsInner = list(kmeans1ResultIterator)
        features_array = [x.features for x in kmeans1ResultsInner]
        kmeans2_list = clustering_for_all_k(features_array)
        kmeans2_statics_df = get_statistics(kmeans2_list, features_array)
        kmeans2_statics_df.to_csv(f"clustering_2/per_label/{predicted_label}.csv")

    print(f"Kmeans2 for all k completed")


# For reference:
# https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
