import csv

from sklearn.cluster import KMeans
import numpy as np

from lego_record import LegoRecord

CSV_DELIMETER = ","
CSV_ARRAY_DELIMETER = "|"
NUM_OF_CLASSES = 432
NUM_OF_STAGE_2_CLUTERS = 2
RESULT_FILENAME = "result.csv"


def data_row_to_record(row: list[str]) -> LegoRecord:
    """
    Get lego record from row read from csv
    """
    feature_strings = row[2].split(CSV_ARRAY_DELIMETER)
    features = [int(feature_str) for feature_str in feature_strings]
    return LegoRecord(path=row[0], label=row[1], features=features)


def load_data_file(filename: str) -> list[LegoRecord]:
    """
    Load list of record as LegoRecord from given csv file
    """
    result = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=CSV_DELIMETER)
        for row in reader:
            result.append(data_row_to_record(row))
    return result


def create_feature_array(record_list: list[LegoRecord]) -> np.ndarray:
    """
    Create numpy array of features from LegoRecord object
    """
    return np.array([record.features for record in record_list])


def perform_clustering(feature_array: np.ndarray, clusters: int):
    """
    Perform clutering on given feature array, return kmeans object
    """
    kmeans = KMeans(n_clusters=clusters, n_init="auto").fit(feature_array)
    return kmeans


def clustering_stage_1(feature_array: np.ndarray) -> np.ndarray:
    kmeans = perform_clustering(feature_array, NUM_OF_CLASSES)
    return kmeans.labels_


def get_features_of_label(
    feature_array: np.ndarray, labels: np.ndarray, label: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get features that were qualified as given label, and their indices in original table
    """
    indices_of_label_2d = np.argwhere(labels == label)
    indices_of_label = np.ravel(indices_of_label_2d)
    features_of_label = [feature_array[i] for i in indices_of_label]
    return features_of_label, indices_of_label


def get_features_by_class(
    feature_array: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generator that returns array of features of subsequent labels, and their indices in original table
    (which is index in csv file as well)
    """
    for label in range(NUM_OF_CLASSES):
        yield get_features_of_label(feature_array, labels, label)


def get_least_popular_label(labels: np.ndarray) -> int:
    """
    Get least popular label from the label list
    """
    histogram, _ = np.histogram(labels, bins=NUM_OF_STAGE_2_CLUTERS)
    least_popular = np.argmin(histogram)
    return least_popular


def get_indices_of_least_popular(labels: np.ndarray, original_indices: np.ndarray):
    """
    Get indicies of features with least popular label. Return indices are indices in original array(based on csv file),
    so that it can be determined, which lego records should be removed
    """
    least_popular_label = get_least_popular_label(labels)
    indices_of_least_popular_2d = np.argwhere(labels == least_popular_label)
    indices_of_least_popular = np.ravel(indices_of_least_popular_2d)
    return [original_indices[i] for i in indices_of_least_popular]


def clustering_stage_2(feature_array: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Iterate over labels and get indices of records that are to be removed (cause they are assigned to least popular label)
    """
    to_remove = np.empty(0)
    for features, indices in get_features_by_class(feature_array, labels):
        kmeans = perform_clustering(features, NUM_OF_STAGE_2_CLUTERS)
        indices_of_least_popular = get_indices_of_least_popular(
            labels=kmeans.labels_, original_indices=indices
        )
        to_remove = np.append(to_remove, indices_of_least_popular)
    return to_remove


def save_result(
    records: list[LegoRecord], labels: np.ndarray, to_remove_indices: np.ndarray
) -> None:
    with open(RESULT_FILENAME, "w") as f:
        for i, record in enumerate(records):
            if i in to_remove_indices:
                continue
            record_with_label_str = f"{str(record)},{str(labels[i])}"
            f.write(record_with_label_str + "\n")


if __name__ == "__main__":
    records = load_data_file("./data.csv")
    feature_array = create_feature_array(records)
    labels = clustering_stage_1(feature_array)
    to_remove_indices = clustering_stage_2(feature_array, labels)
    save_result(records, labels, to_remove_indices)
