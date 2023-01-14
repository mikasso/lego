import os
import shutil
from typing import List

import joblib

from clustering_with_best_k import KMeans2Result

RESULT_IMGS_PATH = "./result/"  # this path has to end with slash


def restore_img_structure(image_path, predicted_label):
    restored_path = os.path.join(os.getcwd(), RESULT_IMGS_PATH, str(predicted_label))
    if not os.path.exists(restored_path):
        os.makedirs(restored_path)
    shutil.copy2(os.path.join(os.getcwd(), image_path), restored_path)  # target filename is /dst/dir/file.ext


def restore_all_imgs(kmeans2_results: List[KMeans2Result], verbose=True):
    for i, result in enumerate(kmeans2_results):
        if verbose == True and i % 1000 == 0:
            print(f"{i+1}/{len(kmeans2_results)}")
        restore_img_structure(result.path, result.predicted_label)


def load_kmeans2_results() -> List[KMeans2Result]:
    return joblib.load("clustering_with_best_k/kmeans_results.d")


if __name__ == "__main__":
    kmeans2_results = load_kmeans2_results()
    shutil.rmtree(RESULT_IMGS_PATH, ignore_errors=True)
    restore_all_imgs(kmeans2_results, True)
