import os
import numpy as np


def get_pathlist(basepath,class_dict):
    data_list = []
    for class_name in class_dict.keys():
        for image_name in os.listdir(os.path.join(basepath, class_name)):
            data_list.append({'image_path': os.path.join(basepath, class_name, image_name),
                              'label_name': class_name,
                              'label_index': class_dict[class_name]})

    np.random.shuffle(data_list)
    return data_list

