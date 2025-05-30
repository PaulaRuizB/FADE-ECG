import torch
import numpy as np
import deepdish as dd
from operator import itemgetter
import os
torch.seed()
import random
random.seed(32)
np.random.seed(32)

class DataGeneratorECG(torch.utils.data.Dataset):
    def __init__(self, path_images, batch_size, sqi, ft, path_datasetinfo, num_channels,
                norm_a_b_max_min, dataset_FT, input_size_seconds,
                 val):
        super().__init__()

        self.path_images = path_images
        self.batch_size = batch_size
        self.sqi = sqi
        self.ft = ft
        self.num_channels = num_channels
        self.val = val
        self.norm_a_b_max_min = norm_a_b_max_min
        self.dataset_FT = dataset_FT
        self.input_size_seconds = input_size_seconds

        print(path_datasetinfo)
        dataset_info = dd.io.load(path_datasetinfo)

        #SQI
        if dataset_FT=="mit_arr_beats":
            self.sqi_values=None
            self.list_samples = dataset_info["list_names"]
        elif dataset_FT=="new_mit_arr_beats" and self.ft and not self.val:
            self.list_samples = dataset_info["list_names_ft"]
            self.labels_x = dataset_info["labels_x_ft"]
            self.labels_y = dataset_info["labels_y_ft"]
            self.positions_normal_signals = [i for i, (x, y) in enumerate(zip(self.labels_x, self.labels_y)) if x == y == "(N"]
            self.list_samples_train = list(itemgetter(*list(self.positions_normal_signals))(self.list_samples))
            print("Samples for FT: {}".format(len(self.list_samples_train)))

        elif dataset_FT == "new_mit_arr_beats" and self.ft and self.val:
            self.list_samples = dataset_info["list_names_train"]
            self.labels_x = dataset_info["labels_x_train"]
            self.labels_y = dataset_info["labels_y_train"]
            self.positions_normal_signals = [i for i, (x, y) in enumerate(zip(self.labels_x, self.labels_y)) if
                                             x == y == "(N"]
            self.list_samples_train = list(itemgetter(*list(self.positions_normal_signals))(self.list_samples))
            print("Samples for validation: {}".format(len(self.list_samples_train)))

        else:
            self.sqi_values = np.array(dataset_info["signal_quality"])
            self.list_samples = dataset_info["list_names"]

        #Images
        if self.ft:
            if self.dataset_FT=="new_mit_arr_beats":
                print("Nothing to do")
            else:
                if self.dataset_FT=="mit_arr":
                    self.sqi_values = dataset_info["signal_quality_new"]
                    self.labels = dataset_info["labels"]
                    self.train_test_label = dataset_info["train_test_label"]

                    self.positions = [idx for idx, (label, sqi, sample) in enumerate(zip(self.train_test_label, self.sqi_values, self.list_samples)) if
                                          label==1 and 0 <=sqi <= self.sqi]

                elif self.dataset_FT == "mit_arr_beats":
                    self.labels = dataset_info["labels"]
                    self.train_test_label = dataset_info["train_test_label"]
                    self.positions = [idx for idx, (label, sample) in enumerate(zip(self.train_test_label, self.list_samples)) if
                                          label==1]
                else:
                    self.positions = [idx for idx, (sample, sqi) in enumerate(zip(self.list_samples, self.sqi_values)) if
                                      sample.startswith('PNSR') and sqi >= self.sqi]

                self.list_samples_train_val = list(itemgetter(*list(self.positions))(self.list_samples))
                random.shuffle(self.list_samples_train_val)
                num = int(len(self.list_samples_train_val) * 1 / 100)

                if not self.val:
                    self.list_samples_train = self.list_samples_train_val[num:]
                    print("Samples for training: {}".format(len(self.list_samples_train)))
                elif self.val:
                    self.list_samples_train = self.list_samples_train_val[:num]
                    print("Samples for validation: {}".format(len(self.list_samples_train)))
        else:
            if not self.val:
                if self.norm_a_b_max_min:
                    info_max_min = dd.io.load(self.path_images + "max3.75_min-1.125_samples_sqi_0.6.h5")

                    self.list_samples = info_max_min["total_samples"]
                    self.list_samples_train = [sample for sample in self.list_samples if not (os.path.split(sample)[-1].startswith('5_'))]
                    random.shuffle(self.list_samples_train)

                    if os.path.exists(self.list_samples[0]):
                        print("Correct paths")
                    print("Samples for training: {}".format(len(self.list_samples_train)))

                else:

                    self.positions = [idx for idx, (sample, sqi) in enumerate(zip(self.list_samples, self.sqi_values)) if
                                  not sample.startswith('5_') and sqi >= self.sqi]
                    self.list_samples_train = list(itemgetter(*list(self.positions))(self.list_samples))
                    random.shuffle(self.list_samples_train)
                    print("Samples for training: {}".format(len( self.list_samples_train)))

            elif self.val:
                if self.norm_a_b_max_min:
                    info_max_min = dd.io.load(self.path_images + "max3.75_min-1.125_samples_sqi_0.6.h5")

                    self.list_samples = info_max_min["total_samples"]
                    self.list_samples_train = [sample for sample in self.list_samples if (os.path.split(sample)[-1].startswith('5_'))]
                    random.shuffle(self.list_samples_train)
                    if os.path.exists(self.list_samples[0]):
                        print("Correct paths")
                    print("Samples for validation: {}".format(len( self.list_samples_train)))
                else:
                    self.positions = [idx for idx, (sample, sqi) in enumerate(zip(self.list_samples, self.sqi_values))
                                      if sample.startswith('5_') and sqi >= self.sqi]
                    self.list_samples_train = list(itemgetter(*list(self.positions))(self.list_samples))
                    random.shuffle(self.list_samples_train)
                    print("Samples for validation: {}".format(len(self.list_samples_train)))

    def __load_sample(self, name_file):
        if self.ft:
            data_i = dd.io.load(self.path_images + self.list_samples_train[name_file]+ ".h5")
        elif self.norm_a_b_max_min:
            data_i = dd.io.load(self.list_samples_train[name_file])
        else:
            data_i = dd.io.load(self.path_images + self.list_samples_train[name_file]+ ".h5")


        if type(data_i) is dict:
            return data_i['input'], data_i['label']
        elif self.dataset_FT == "mit_arr" or self.dataset_FT == "mit_arr_beats":
            return data_i[0, int((4-self.input_size_seconds)*128):], data_i[1, int((4-self.input_size_seconds)*128):]
        elif self.dataset_FT == "new_mit_arr_beats":
            return data_i[0,:], data_i[1,:]
        else:
            if data_i.shape[1]==256:
                return data_i[:self.num_channels], data_i[6:6+self.num_channels]
            elif data_i.shape[1]==512:
                return data_i[:self.num_channels, int((4-self.input_size_seconds)*128):], data_i[6:6+self.num_channels, int((4-self.input_size_seconds)*128):]
            else:
                data_i = np.transpose(data_i, (1, 0))
                return data_i[:self.num_channels], data_i[6:6+self.num_channels]

    def __len__(self):
        return len(self.list_samples_train)

    def __getitem__(self, idx):

        sample, label = self.__load_sample(idx)

        if self.norm_a_b_max_min:

            a_g = 0
            b_g = 2.5
            a_p = -0.75
            b_p = 0

            if self.ft:
                if self.dataset_FT == "mit_arr":
                    max_g = 3.5
                    min_p = -3.5
                else:
                    max_g = 1
                    min_p = -0.5
            else:
                max_g = 2.5
                min_p = -0.75

            min_g = 0
            max_p = 0

            sample = torch.clip(torch.tensor(sample), min=min_p, max=max_g)
            label = torch.clip(torch.tensor(label), min=min_p, max=max_g)
            sample[sample > 0] = a_g + (((sample[sample > 0] - min_g) * (b_g - a_g)) / (max_g - min_g))
            label[label > 0] = a_g + (((label[label > 0] - min_g) * (b_g - a_g)) / (max_g - min_g))

            sample[sample < 0] = a_p + (((sample[sample < 0] - min_p) * (b_p - a_p)) / (max_p - min_p))
            label[label < 0] = a_p + (((label[label < 0] - min_p) * (b_p - a_p)) / (max_p - min_p))

            if self.dataset_FT== "mit_arr":
                label = np.expand_dims(label, axis=0)
                sample = np.expand_dims(sample, axis=0)

            return sample, label
        else:
            if self.dataset_FT== "mit_arr" or self.dataset_FT== "mit_arr_beats" or self.dataset_FT== "new_mit_arr_beats":
                label = np.expand_dims(label, axis=0)
                sample = np.expand_dims(sample, axis=0)
            return sample, label
