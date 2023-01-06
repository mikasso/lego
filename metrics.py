import re
import os

def get_max_label_percentage(path):
    """
    calculates what part of images in dir are those of the most numerous class
    """
    regex = r".*_"
    images = os.listdir(path)
    image_labels=[re.search(regex, image).group(0)[:-1] for image in images]
    labels_count={}
    for label in image_labels:
        if label in labels_count.keys():
            labels_count[label]+=1
        else:
            labels_count[label]=1
    #print(max(labels_count, key=stats.get)) label of max value
    return max(labels_count.values())/len(images)


def print_metrics():
    """
 
    """
    labels = os.listdir("./result")
    mean=sum([get_max_label_percentage("./result/"+label) for label in labels])/len(labels)
    print("Avg prevalence of the most numerous class: {}".format(mean))
    
    
if __name__ == "__main__":
    print_metrics()
