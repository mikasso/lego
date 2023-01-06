import os
from typing import List, Tuple

import joblib
import numpy as np


def read_pickle(pickle_filepath: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
    features, labels, paths = joblib.load(pickle_filepath)
    paths = list(map(lambda path: path.replace("\\", "/"), paths))
    labels = list(map(lambda label: str(label), labels))
    return features, labels, paths


def read_features(features_dir: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
    features, labels, paths = [], [], []
    files = os.listdir(features_dir)
    pickles = list(filter(lambda f: f.endswith(".pkl"), files))
    for pickle in pickles:
        features_part, labels_part, paths_part = read_pickle(os.path.join(features_dir, pickle))
        paths.extend(paths_part)
        labels.extend(labels_part)
        features.extend(features_part)
    return features, labels, paths


def read_features_as_np(features_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, paths = read_features(features_dir)
    return np.array(features), np.array(labels), np.array(paths)


def read_labels(features_dir: str) -> np.ndarray:
    return joblib.load(os.path.join(features_dir, "labels.np"))
