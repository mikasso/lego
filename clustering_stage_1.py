from collections import Counter
from dataclasses import dataclass
import dataclasses
import itertools
import operator
from typing import Generator, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from read_features import read_features_as_np

NUM_OF_LABELS = 432


@dataclass
class KMeansStatistic:
    predicted_label: int
    dominant_label: str
    count_of_all_items_assigned: int
    dominant_occurences: int
    expected_k: int

    def __init__(self, predicted_label: str, original_labels: List[str]):
        count_of_all_items = len(original_labels)
        original_labels_counter = Counter(original_labels)
        dominant_label, dominant_occurences = original_labels_counter.most_common(1)[0]
        self.dominant_label = dominant_label
        self.count_of_all_items_assigned = count_of_all_items
        self.dominant_occurences = dominant_occurences
        self.predicted_label = predicted_label
        self.expected_k = len(list(original_labels_counter))


def iterate_results_by_predicted(
    result_tuples: List[Tuple[int, str, np.ndarray, str]]
) -> Generator[Tuple[int, List[Tuple[int, str, np.ndarray, str]]], None, None]:
    keyfunc = operator.itemgetter(0)
    result_tuples = sorted(result_tuples, key=keyfunc)
    it = itertools.groupby(result_tuples, key=keyfunc)
    for predicted_label, tuples_it_for_predicted in it:
        yield predicted_label, list(tuples_it_for_predicted)


def clustering_stage_1(feature_array: np.ndarray) -> KMeans:
    kmeans = KMeans(n_clusters=NUM_OF_LABELS, n_init="auto").fit(feature_array)
    return kmeans


def get_kmeans_stats(result_tuples: List[Tuple[int, str, np.ndarray, str]]) -> List[KMeansStatistic]:
    for predicted_label, tuples_for_predicted in iterate_results_by_predicted(result_tuples):
        yield KMeansStatistic(predicted_label, [x[1] for x in tuples_for_predicted])


def kmeans_stats_to_df(kmneans_stats: List[KMeansStatistic]) -> pd.DataFrame:
    """
    Return dataframe from kmneans_stats list
    predicted label - aka folder after first clustering
    dominant original label - true label which is dominant in the folder
    dominant occurences
    samples - number of samples used during clustering
    k_expected - is also actual number of unique true labels
    """
    return pd.DataFrame.from_records(
        [dataclasses.astuple(item) for item in kmneans_stats],
        columns=["predicted label", "dominant orignal label", "samples", "dominant occurences", "k_expected"],
    )


@dataclass
class KMeans1Result:
    predicted_label: int
    original_label: str
    features: np.ndarray
    path: str


def read_kmeans1_results(path: str) -> List[KMeans1Result]:
    kMeans1Results = joblib.load(path)
    return kMeans1Results


if __name__ == "__main__":
    print("Loading data..")
    input_features, input_original_labels, input_paths = read_features_as_np("features_extracted")
    print("Data loaded")

    print("Running kmeans 1")
    kmeans1 = clustering_stage_1(input_features)
    print("Kmeans1 completed")
    print(f"Kmeans1 inertia: {kmeans1.inertia_}")

    predicted_labels_1 = kmeans1.labels_
    kmeans1_result_tuples = list(zip(predicted_labels_1, input_original_labels, input_features, input_paths))
    kmeans1_result_tuples = sorted(kmeans1_result_tuples, key=operator.itemgetter(0))

    print(f"Gathering stats")
    kmeans1_stats = list(get_kmeans_stats(kmeans1_result_tuples))
    df_kmeans1_stats = kmeans_stats_to_df(kmeans1_stats)
    print(df_kmeans1_stats)

    print("Saving Results")

    kMeans1Results = [KMeans1Result(x[0], x[1], x[2], x[3]) for x in kmeans1_result_tuples]
    joblib.dump(kMeans1Results, "clustering_1/kmeans1_results.d")
    df_kmeans1_stats.to_csv("clustering_1/kmeans_stats.csv")

    f = open("clustering_1/inertia.txt", "w")
    f.write(f"Inertia = {kmeans1.inertia_}")
    f.close()
