import os
import shutil

import joblib


def restore_img_structure(image_path, predicted_label):
    result_path = "result/"  # this path has to end with slash
    restored_path = os.path.join(os.getcwd(), result_path, str(predicted_label))
    if not os.path.exists(restored_path):
        os.makedirs(restored_path)
    shutil.copy2(os.path.join(os.getcwd(), image_path), restored_path)  # target filename is /dst/dir/file.ext


def restore_all_imgs(result_tuples, verbose=True):
    shutil.rmtree("./result", ignore_errors=True)
    for i, (kmeans1_label, kmeans2_label, label_original, feature, path) in enumerate(result_tuples):
        if verbose == True and i % 1000 == 0:
            print(f"{i+1}/{len(result_tuples)}")
        restore_img_structure(path, kmeans1_label)


if __name__ == "__main__":
    kmeans2_result_tuples = joblib.load("clustering_results/kmeans2.d")
    restore_all_imgs(kmeans2_result_tuples, True)
