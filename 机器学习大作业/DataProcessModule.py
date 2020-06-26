import numpy as np
import os
from PIL import Image

dir_to_index = {"front": 0, "up": 1, "left": 2, "right": 3}
index_to_dir = {0 : "front", 1 : "up", 2 : "left", 3 : "right"}
name_to_index = {'2019051013':0,"2019051014":1,"2019051018":2,"2019051019":3,"2019051023":4
,"2019051026":5,"2019052031":6,"2019052054":7,"2019052059":8,
"2019053073":9,"2019053075":10}
index_to_name = {0:'2019051013',1:"2019051014",2:"2019051018",3:"2019051019",4:"2019051023"
,5:"2019051026",6:"2019052031",7:"2019052054",8:"2019052059",
9:"2019053073",10:"2019053075"}


class ImageData:
    def __load_data(self, data_dir, is_binarization):
        __datas = []
        __name_labels = []
        __dir_labels = []
        __fpaths = []
        for fname in os.listdir(data_dir):
            __fpath = os.path.join(data_dir, fname)
            __image = Image.open(__fpath)
            if is_binarization:
                __data = np.array(__image.convert("1"), dtype= "int64").flatten()
                # Image._show(__image.convert("1"))
            else:
                __data = np.array(__image) / 255.0
            __name_label = fname.split("_")[0]
            __dir_label = fname.split("_")[1].split(".")[0]
            __datas.append(__data)
            __name_labels.append(name_to_index[__name_label])
            __dir_labels.append(dir_to_index[__dir_label])
        datas = np.array(__datas)
        name_labels = np.array(__name_labels)
        dir_labels = np.array(__dir_labels)
        return datas, name_labels, dir_labels
    
    def __init__(self, file_name, is_shuffle, is_binarization):
        __datas, __name_labels, __dir_labels = self.__load_data(file_name, is_binarization)
        self.data = __datas
        self.name_labels = __name_labels 
        self.dir_labels = __dir_labels
        self.__indicator = 0
        self.__num_examples = self.data.shape[0]
        self.__is_shuffle = is_shuffle
        if is_shuffle:
            self.__shuffle_data()

    def __shuffle_data(self):
        p = np.random.permutation(self.data.shape[0])
        self.data = self.data[p]
        self.name_labels = self.name_labels[p]
        self.dir_labels = self.dir_labels[p]
    
    def next_batch(self, batch_size, is_cnn, is_bp_name_lebals):
        end_indictor = self.__indicator + batch_size
        if end_indictor > self.__num_examples: ## finish a epho
            if self.__is_shuffle:
                self.__shuffle_data()
                self.__indicator = 0
                end_indictor = batch_size
            else:
                raise Exception("have no more examples")
        if end_indictor > self.__num_examples:
            raise Exception("batch size is larger than all examples")
      
        batch_data = self.data[self.__indicator:end_indictor]
        batch_name_lebals = self.name_labels[self.__indicator:end_indictor]
        batch_dir_lebals = self.dir_labels[self.__indicator:end_indictor]
        self.__indicator = end_indictor
        if is_cnn:
            return batch_data,batch_name_lebals,batch_dir_lebals
        else:
            if is_bp_name_lebals:
                return batch_data ,batch_name_lebals
            else:
                return batch_data, batch_dir_lebals


if __name__ == "__main__":
    data_dir = "/media/test/B/python_文件/机器学习大作业"
    train_filename = os.path.join(data_dir,"train")
    test_filename = os.path.join(data_dir,"test")
    train_data = ImageData(train_filename,True, True)
    print("BP_train_data load success")
    test_data = ImageData(test_filename,False, True)
    print("test_data load success")