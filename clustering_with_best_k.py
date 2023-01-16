from collections import Counter
from dataclasses import dataclass
import itertools
from typing import List
import joblib
import numpy as np

import pandas as pd
from sklearn.cluster import KMeans
from clustering_stage_1 import (
    NUM_OF_LABELS,
    KMeans1Result,
    KMeansStatistic,
    kmeans_stats_to_df,
    read_kmeans1_results,
)
from clustering_stage_2 import get_kMeans1Results_iterator_grouped_by_predicted_label
import matplotlib.pyplot as plt


@dataclass
class KMeans2Result:
    """
    predicted_label - label after first kmeans - aka folder
    """

    predicted_label: int
    """
    predicted_label_2 - label after second kmenas, in general not used for statistics but needed for internal uses cutting etc.
    """
    predicted_label_2: int
    original_label: str
    features: np.ndarray
    path: str


def get_filtered_result(
    best_kmeans: KMeans, results: List[KMeans2Result]
) -> List[KMeans2Result]:
    most_common_kmeans2_label, _ = Counter(best_kmeans.labels_).most_common(1)[0]
    filtered_results = list(
        filter(
            lambda result: result.predicted_label_2 == most_common_kmeans2_label,
            results,
        )
    )
    return filtered_results


def get_kmeans2_results(
    predicted_label: str,
    predicted_labels_2: List[str],
    original_labels: np.ndarray,
    features_array: np.ndarray,
    paths: np.ndarray,
) -> List[KMeans2Result]:
    for predicted_label_2, original_label, features, path in zip(
        predicted_labels_2, original_labels, features_array, paths
    ):
        yield KMeans2Result(
            predicted_label, predicted_label_2, original_label, features, path
        )


def load_statistics(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def silhouette_metric(df: pd.DataFrame) -> int:
    index = np.argmax(df["silhouette_score"])
    return df["k"][index]


def inertia_metric(df: pd.DataFrame) -> int:
    dif = pd.Series.to_numpy(df["inertia"].diff()[1:])
    max_decrease = dif.min()
    desired_decrease = max_decrease * 0.1
    decreasing_too_much = np.argwhere(dif < desired_decrease)
    last_with_decrease = decreasing_too_much[-1][0]
    return df["k"][last_with_decrease]


def find_best_k(df: pd.DataFrame, metric="silhouette_score") -> int:
    # df columns = ["k", "inertia", "silhouette_score", "biggest_cluster_sizes", "samples_count"]
    if metric == "silhouette_score":
        return silhouette_metric(df)
    if metric == "inertia":
        return inertia_metric(df)
    if metric == "something_between":
        return (inertia_metric(df) + silhouette_metric(df)) // 2
    else:
        raise ValueError


if __name__ == "__main__":
    print("Loading kmeans1 results..")
    kMeans1Results = read_kmeans1_results("clustering_1/kmeans1_results.d")
    print("Data loaded")

    print("Running clustering with best k")

    stats_list = []
    used_k_list = []
    results_list = []

    iterator = get_kMeans1Results_iterator_grouped_by_predicted_label(kMeans1Results)
    for predicted_label, kmeans1ResultIterator in iterator:
        print(f"Runnig kmeans with best k: {predicted_label}/{NUM_OF_LABELS}")
        kmeans1ResultsInner = list(kmeans1ResultIterator)
        original_labels = [x.original_label for x in kmeans1ResultsInner]
        features_array = [x.features for x in kmeans1ResultsInner]
        paths = [x.path for x in kmeans1ResultsInner]

        kmeans2_statics_df = load_statistics(
            f"clustering_2/per_label/{predicted_label}.csv"
        )
        best_k = find_best_k(kmeans2_statics_df, metric="inertia")
        best_kmeans = KMeans(n_clusters=best_k, n_init="auto").fit(features_array)

        # Filter out results, keep only for dominant original label
        results = get_kmeans2_results(
            predicted_label, best_kmeans.labels_, original_labels, features_array, paths
        )
        filtered_results = get_filtered_result(best_kmeans, list(results))
        filtered_original_labels = [x.original_label for x in filtered_results]

        # Save stats and results
        results_list.extend(filtered_results)
        stats_list.append(KMeansStatistic(predicted_label, filtered_original_labels))
        used_k_list.append(best_k)

    print("Clustering with best k completed")

    print("Gathering statistics")
    df_stats = kmeans_stats_to_df(stats_list)
    df_stats["used_k"] = used_k_list

    print(f"Saving results")
    joblib.dump(results_list, "clustering_with_best_k/kmeans_results.d")
    df_stats.to_csv("clustering_with_best_k/kmeans_stats.csv")
