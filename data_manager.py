import numpy as np
import os, os.path
import cv2 as cv
from sklearn.model_selection import train_test_split

#
class DataManagerDeconvolution():
    """
    Manages data laoding, train, test split and normalization
    """
    def __init__(self):
        self.i = 0

    def load_to_file(self, img_path, mask_path):
        _, _, files = next(os.walk(img_path))
        file_count = len(files)

        for img_number in range(1, file_count + 1):
            path_img = '{}{}.jpg'.format(img_path, img_number)
            path_mask = '{}{}.jpg'.format(mask_path, img_number)
            img = cv.imread(path_img)
            mask = cv.imread(path_mask)

            if img_number == 1:
                img_data = np.expand_dims(img, axis=0)
                mask_data = np.expand_dims(mask, axis=0)
            else:
                img = np.expand_dims(img, axis=0)
                mask = np.expand_dims(mask, axis=0)

                img_data = np.append(img_data, img, axis=0)
                mask_data = np.append(mask_data, mask, axis=0)

            if img_number % 100 == 0:
                print('Loaded {} %'.format(int(img_number / file_count * 100)))

        np.save('img_data', img_data)
        np.save('mask_data', mask_data)
        print('Data saved to img_data.npy and mask_data.npy')

    def load_from_file(self):
        self.img_data = np.load('img_data.npy')
        self.mask_data = np.load('mask_data.npy')

    def get_img(self, index):
        return self.img_data[index]

    def get_mask(self, index):
        return self.mask_data[index]

    def set_up(self):
        #normalization
        self.img_data_norm = self.img_data / 255
        self.mask_data_norm = self.mask_data / 255

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.img_data_norm, self.mask_data_norm, test_size = 0.33, random_state = 42)
        print("Data is Set Up")
    def next_batch(self, batch_size):
        x = self.X_train[self.i:self.i + batch_size].reshape(batch_size,40,40,3)
        y = self.y_train[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.X_train)
        return x, y

class DataManagerConvolution():
    """
    Manages data laoding, train, test split and normalization
    """
    def __init__(self):
        self.i = 0

    def load_to_file(self, img_path):
        _, _, files = next(os.walk(img_path))
        file_count = len(files)

        for img_number in range(1, file_count + 1):
            path_img = '{}{}.jpg'.format(img_path, img_number)
            img = cv.imread(path_img)

            if img_number == 1:
                img_data = np.expand_dims(img, axis=0)
            else:
                img = np.expand_dims(img, axis=0)

                img_data = np.append(img_data, img, axis=0)

            if img_number % 100 == 0:
                print('Loaded {} %'.format(int(img_number / file_count * 100)))

        np.save('img_0_data', img_data)
        print('Data saved to img_data.npy')

    def load_from_file(self):
        self.img_data = np.load('img_data.npy')
        self.img_0_data = np.load('img_0_data.npy')

        true_label = np.float32([[1.0, 0.0]])
        true_label = np.repeat(true_label, self.img_data.shape[0], axis=0)

        false_label = np.float32([[0.0, 1.0]])
        false_label = np.repeat(false_label, self.img_0_data.shape[0], axis=0)

        self.img_data = np.concatenate((self.img_data, self.img_0_data), axis=0)

        self.label = np.concatenate((true_label, false_label), axis=0)

    def get_img(self, index):
        return self.img_data[index]

    def get_label(self, index):
        return self.label[index]

    def set_up(self):
        #normalization
        self.img_data_norm = self.img_data / 255

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.img_data_norm, self.label, test_size = 0.33, random_state = 42)
        print("Data is Set Up")
    def next_batch(self, batch_size):
        x = self.X_train[self.i:self.i + batch_size].reshape(batch_size,40,40,3)
        y = self.y_train[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.X_train)
        return x, y