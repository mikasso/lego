from collections import Counter
from dataclasses import dataclass
import dataclasses
import itertools
from math import sqrt
import operator
import pandas as pd
from typing import Generator, List, Tuple
import joblib
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
import numpy as np
from clustering_stage_1 import KMeans1Result, KMeansStatistic, kmeans_stats_to_df

from matplotlib import pyplot as plt


NUM_OF_LABELS = 432


def clustering_for_all_k(features: np.ndarray) -> List[KMeans]:
    kmeans_list = []
    max_clusters = min(len(features), NUM_OF_LABELS)
    for clusters in range(2, max_clusters):
        kmeans = KMeans(n_clusters=clusters, n_init="auto").fit(features)
        kmeans_list.append(kmeans)

    return kmeans_list


def get_result_histogram(label_pairs):
    labels_dict = {}
    for pair in label_pairs:
        old_label = pair[1]
        new_label = pair[0]
        existing_set = labels_dict.get(old_label)
        if existing_set is None:
            labels_dict[old_label] = {new_label}
        else:
            existing_set.add(new_label)
    sizes = [len(labels) for labels in labels_dict.values()]
    sizes_np = np.asarray(sizes)
    max_size = np.amax(sizes_np)
    histogram, _ = np.histogram(sizes_np, bins=max_size)
    return histogram


def show_histogram(filtered_pairs):
    histogram = get_result_histogram(filtered_pairs)
    plt.bar(range(0, len(histogram)), histogram)
    plt.title("How many orignal labels are assigned to new labels")
    plt.show()


def min_by_custom(kmeans: KMeans) -> float:
    _, biggest_cluster_size = Counter(kmeans.labels_).most_common(1)[0]
    samples_count = kmeans.cluster_centers_.shape[0]
    return kmeans.inertia_ * kmeans.inertia_ / sqrt((biggest_cluster_size / samples_count))


@dataclass
class KMeans2Result:
    predicted_label: int
    predicted_label_2: int
    original_label: str
    features: np.ndarray
    path: str


def get_filtered_result(best_kmeans: KMeans, results: List[KMeans2Result]) -> List[KMeans2Result]:
    most_common_kmeans2_label, _ = Counter(best_kmeans.labels_).most_common(1)[0]
    filtered_results = list(filter(lambda result: result.predicted_label_2 == most_common_kmeans2_label, results))
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
        yield KMeans2Result(predicted_label, predicted_label_2, original_label, features, path)


def read_kmeans1_results(path: str) -> List[KMeans1Result]:
    """
    Loads and returns tuple of (predicted labels, original_labels, features, paths)
    """
    kMeans1Results = joblib.load(path)
    return kMeans1Results


def get_statistics(kmeans_list: List[KMeans], features_array: np.ndarray) -> pd.DataFrame:
    inertias = []
    k_values = []
    silhouette_scores = []
    adjusted_silhouette_scores = []
    adjusted_inertias = []
    for kmeans in kmeans_list:
        k_values.append(kmeans.n_clusters)
        inertias.append(kmeans.inertia_)
        silhouette_score_value = silhouette_score(features_array, kmeans.labels_, metric="euclidean")
        silhouette_scores.append(silhouette_score_value)
        _, biggest_cluster_size = Counter(kmeans.labels_).most_common(1)[0]
        samples_count = len(features_array)
        ratio = pow(float(biggest_cluster_size) / samples_count, 2)
        adjusted_silhouette_scores.append(silhouette_score_value / ratio)
        adjusted_inertias.append(kmeans.inertia_ / ratio)

    return pd.DataFrame(
        {
            "k": k_values,
            "inertia": inertias,
            "silhouette_score": silhouette_scores,
            "adj_inertia": adjusted_inertias,
            "adj_silhouette_score": adjusted_silhouette_scores,
        }
    )


def find_best_k(df: pd.DataFrame, metric="silhouette_score") -> int:
    row = df[df[metric] == df[metric].min()]
    return row["k"].iloc[0]


if __name__ == "__main__":
    print("Loading kmeans1 results..")
    kMeans1Results = read_kmeans1_results("clustering_1/kmeans1_results.d")
    print("Data loaded")

    print("Running kmeans2")

    kmeans2_stats = []
    kmeans2_opt = []
    kmeans2_result_tuples = []
    kmeans2_statics_df_list = []

    keyfunc = lambda x: x.predicted_label
    result_tuples = sorted(kMeans1Results, key=keyfunc)
    it = itertools.groupby(kMeans1Results, key=keyfunc)

    for predicted_label, kmeans1ResultIterator in it:
        print(f"Runnig kmeans2 stage: {predicted_label}/{NUM_OF_LABELS}")
        kmeans1ResultsInner = list(kmeans1ResultIterator)
        original_labels = [x.original_label for x in kmeans1ResultsInner]
        features_array = [x.features for x in kmeans1ResultsInner]
        paths = [x.path for x in kmeans1ResultsInner]

        kmeans2_list = clustering_for_all_k(features_array)

        kmeans2_statics_df = get_statistics(kmeans2_list, features_array)
        best_k = find_best_k(kmeans2_statics_df)
        best_kmeans = next(kmeans2 for kmeans2 in kmeans2_list if kmeans2.n_clusters == best_k)

        # Filter out results, keep only for dominant original label
        results = get_kmeans2_results(predicted_label, best_kmeans.labels_, original_labels, features_array, paths)
        filtered_results = get_filtered_result(best_kmeans, list(results))
        filtered_original_labels = [x.original_label for x in filtered_results]

        # Save results from kmeans2 stage
        kmeans2_result_tuples.extend(filtered_results)
        kmeans2_stats.append(KMeansStatistic(predicted_label, filtered_original_labels))
        kmeans2_opt.append(tuple([best_kmeans.cluster_centers_.shape[0], best_kmeans.inertia_]))
        kmeans2_statics_df_list.append(kmeans2_statics_df)
        break

    print(f"Kmeans2 completed")

    df_kmeans2 = kmeans_stats_to_df(kmeans2_stats)
    print(df_kmeans2)

    df_kmeans_opt = pd.DataFrame.from_records(
        [item for item in kmeans2_opt],
        columns=["k", "inertia"],
    )
    print(df_kmeans_opt)

    print(f"Saving results")
    joblib.dump(kmeans2_result_tuples, "clustering_2/kmeans2.d")
    df_kmeans_opt.to_csv("clustering_2/kmenas_opt.csv")
    df_kmeans2.to_csv("clustering_2/kmeans2.csv")
    # can do that because idx == predicted_label
    for idx, df in enumerate(kmeans2_statics_df_list):
        df.to_csv(f"clustering_2/per_label/{idx}.csv")


# analiza jakosci 1 kminsa <- tez moze byc histogram
# dla drugiego kmeans sprobowac minializowac odleglosci i dla tego k wziac wiekszy zbiór, k będzie inne dla kazedego folderu
# statystki dla zmniejszenia zbiorow
# na piatek tylko wyniki
# sillouhete coefficient zamiast inertia
# https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
