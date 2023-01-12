from collections import Counter
from dataclasses import dataclass
import dataclasses
import itertools
from math import sqrt
import operator
import pandas as pd
from typing import Generator, List, Tuple
import joblib

from sklearn.cluster import KMeans
import numpy as np

from read_features import read_features_as_np
from matplotlib import pyplot as plt


NUM_OF_LABELS = 432


def perform_clustering(feature_array: np.ndarray, clusters: int) -> KMeans:
    """
    Perform clutering on given feature array, return kmeans object
    """
    kmeans = KMeans(n_clusters=clusters, n_init="auto").fit(feature_array)
    return kmeans


def clustering_stage_1(feature_array: np.ndarray) -> KMeans:
    kmeans = perform_clustering(feature_array, NUM_OF_LABELS)
    return kmeans


def clustering_by_min_criteria(features: np.ndarray, criteria) -> KMeans:
    kmeans_list = []
    max_clusters = min(len(features), int((len(features) * 0.9) + 1))
    for clusters in range(2, max_clusters):
        kmeans = perform_clustering(features, clusters)
        kmeans_list.append(kmeans)

    opt_kmeans = min(kmeans_list, key=criteria)
    return opt_kmeans


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


@dataclass
class KMeansStatistic:
    predicted_label: int
    dominant_label: str
    count_of_all_items_assigned: int
    dominant_occurences: int

    def __init__(self, predicted_label: str, original_labels: List[str]):
        count_of_all_items = len(original_labels)
        dominant_label, dominant_occurences = Counter(original_labels).most_common(1)[0]
        self.dominant_label = dominant_label
        self.count_of_all_items_assigned = count_of_all_items
        self.dominant_occurences = dominant_occurences
        self.predicted_label = predicted_label


def iterate_results_by_predicted(
    result_tuples: List[Tuple[int, str, np.ndarray, str]]
) -> Generator[Tuple[int, List[Tuple[int, str, np.ndarray, str]]], None, None]:
    keyfunc = operator.itemgetter(0)
    result_tuples = sorted(result_tuples, key=keyfunc)
    it = itertools.groupby(result_tuples, key=keyfunc)
    for predicted_label, tuples_it_for_predicted in it:
        yield predicted_label, list(tuples_it_for_predicted)


def get_kmeans_stats(result_tuples: List[Tuple[int, str, np.ndarray, str]]) -> List[KMeansStatistic]:
    for predicted_label, tuples_for_predicted in iterate_results_by_predicted(result_tuples):
        yield KMeansStatistic(predicted_label, [x[1] for x in tuples_for_predicted])


def kmeans_stats_to_df(kmneans_stats: List[KMeansStatistic]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [dataclasses.astuple(item) for item in kmneans_stats],
        columns=["predicted label", "dominant orignal label", "all items", "dominant occurences"],
    )


def min_by_inertia(kmeans: KMeans) -> float:
    return kmeans.inertia_


def min_by_custom(kmeans: KMeans) -> float:
    _, biggest_cluster_size = Counter(kmeans.labels_).most_common(1)[0]
    samples_count = kmeans.cluster_centers_.shape[0]
    return kmeans.inertia_ * kmeans.inertia_ / sqrt((biggest_cluster_size / samples_count))


if __name__ == "__main__":
    print("Loading data..")
    input_features, input_original_labels, input_paths = read_features_as_np("features_extracted")
    print("Data loaded")

    print("Running kmeans 1")
    kmeans1 = clustering_stage_1(input_features)
    print("Kmeans1 completed")

    print("Running kmeans2")
    predicted_labels_1 = kmeans1.labels_
    kmeans1_result_tuples = list(zip(predicted_labels_1, input_original_labels, input_features, input_paths))

    kmeans2_stats = []
    kmeans2_opt = []
    kmeans2_result_tuples = []
    to_iter = list(iterate_results_by_predicted(kmeans1_result_tuples))[0:3]
    iterations = len(to_iter)
    for idx, (predicted_label, tuples_for_predicted) in enumerate(to_iter):
        print(f"Runnig kmeans2 stage: {idx}/{iterations}")
        original_labels = [x[1] for x in tuples_for_predicted]
        features = [x[2] for x in tuples_for_predicted]
        paths = [x[3] for x in tuples_for_predicted]
        kmeans2 = clustering_by_min_criteria(features, criteria=min_by_inertia)
        # Filter out results, keep only for dominant original label
        results = zip([predicted_label] * len(paths), kmeans2.labels_, original_labels, features, paths)
        most_common_kmeans2_label, _ = Counter(kmeans2.labels_).most_common(1)[0]
        filtered_results = list(filter(lambda result: result[1] == most_common_kmeans2_label, results))
        filtered_original_labels = [x[2] for x in filtered_results]
        # Save results from kmeans2 stage
        kmeans2_result_tuples.extend(filtered_results)
        kmeans2_stats.append(KMeansStatistic(predicted_label, filtered_original_labels))
        kmeans2_opt.append(tuple([kmeans2.cluster_centers_.shape[0], kmeans2.inertia_]))

    print(f"Kmeans2 completed")

    print(f"Gathering stats")
    kmeans1_stats = list(get_kmeans_stats(kmeans1_result_tuples))
    df_kmeans1 = kmeans_stats_to_df(kmeans1_stats)
    print(df_kmeans1)

    df_kmeans2 = kmeans_stats_to_df(kmeans2_stats)
    print(df_kmeans2)

    df_kmeans_opt = pd.DataFrame.from_records(
        [item for item in kmeans2_opt],
        columns=["k", "inertia"],
    )
    print(df_kmeans_opt)

    print(f"Saving results")
    joblib.dump(kmeans1_result_tuples, "clustering_results/kmeans1.d")
    joblib.dump(kmeans2_result_tuples, "clustering_results/kmeans2.d")
    df_kmeans_opt.to_csv("clustering_results/kmenas_opt.csv")
    df_kmeans1.to_csv("clustering_results/kmeans1.csv")
    df_kmeans2.to_csv("clustering_results/kmeans2.csv")


# analiza jakosci 1 kminsa <- tez moze byc histogram
# dla drugiego kmeans sprobowac minializowac odleglosci i dla tego k wziac wiekszy zbiór, k będzie inne dla kazedego folderu
# statystki dla zmniejszenia zbiorow
# na piatek tylko wyniki
