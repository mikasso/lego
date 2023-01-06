import os
import shutil


def restore_img_structure(label, image_path, predictedClass):
    result_path="result/"#this path has to end with slash
    restored_path=os.path.join(os.getcwd(),result_path,str(predictedClass) )
    print (restored_path)
    if not os.path.exists(restored_path):
        os.makedirs(restored_path)
    shutil.copy2(os.path.join(os.getcwd(),image_path), restored_path) # target filename is /dst/dir/file.ext
    
        
if __name__ == "__main__":
    restore_img_structure(9,"test/0lDm_Bs5_1620809531797.jpg",126)
    assert os.path.exists("/home/ian/iui/result/126/0lDm_Bs5_1620809531797.jpg")==True    
