import os
import shutil


def restore_img_structure(image_path, predicted_label):
    result_path = "result/"  # this path has to end with slash
    restored_path = os.path.join(os.getcwd(), result_path, str(predicted_label))
    if not os.path.exists(restored_path):
        os.makedirs(restored_path)
    shutil.copy2(os.path.join(os.getcwd(), image_path), restored_path)  # target filename is /dst/dir/file.ext


def restore_all_imgs(filtered_pairs, verbose=True):
    shutil.rmtree("./result")
    for i, (label, label_original, path) in enumerate(filtered_pairs):
        if verbose == True and i % 1000 == 0:
            print(f"{i+1}/{len(filtered_pairs)}")
        restore_img_structure(path, label)


if __name__ == "__main__":
    restore_img_structure(9, "test/0lDm_Bs5_1620809531797.jpg", 126)
    assert os.path.exists("/home/ian/iui/result/126/0lDm_Bs5_1620809531797.jpg") == True
