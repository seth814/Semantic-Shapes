import numpy as np
import pickle
import os
from PIL import Image
from config import imshape, n_classes, labels, model_name
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import json
import tensorflow as tf


ia.seed(1)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Multiply((1.2, 1.5)),
    iaa.Affine(
        #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        rotate=(-90, 90)
    ),
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 8))
    )
], random_order=True)


class DataGenerator(tf.keras.utils.Sequence):
    # Generates data for Keras
    def __init__(self, image_paths, annot_paths, batch_size=32,
                 shuffle=True, augment=False):
        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()


    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size))


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        image_paths = [self.image_paths[k] for k in indexes]
        annot_paths = [self.annot_paths[k] for k in indexes]

        X, y = self.__data_generation(image_paths, annot_paths)

        return X, y


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def get_poly(self, annot_path):
        # reads in shape_dicts
        with open(annot_path) as handle:
            data = json.load(handle)

        shape_dicts = data['shapes']

        return shape_dicts


    def create_binary_masks(self, im, shape_dicts):
        # image must be grayscale
        blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)

        for shape in shape_dicts:
            if shape['label'] != 'background':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(blank, [points], 255)
        blank = blank / 255.0

        return np.expand_dims(blank, axis=2)


    def create_multi_masks(self, im, shape_dicts):

        channels = []
        cls = [x['label'] for x in shape_dicts]
        poly = [np.array(x['points'], dtype=np.int32) for x in shape_dicts]
        label2poly = dict(zip(cls, poly))
        background = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)

        # iterate through objects of interest
        for i, label in enumerate(labels):

            blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)

            if label in cls:
                cv2.fillPoly(blank, [label2poly[label]], 255)
                cv2.fillPoly(background, [label2poly[label]], 255)

            channels.append(blank)

        # handle an image where only background is present
        if 'background' in cls:
            background = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
            cv2.fillPoly(background, [label2poly['background']], 255)
        else:
            _, background = cv2.threshold(background, 127, 255, cv2.THRESH_BINARY_INV)
        channels.append(background)

        Y = np.stack(channels, axis=2) / 255.0

        return Y


    def augment_poly(self, im, shape_dicts):
        # augments an image and it's polygons

        points = []
        aug_shape_dicts = []
        i = 0

        for shape in shape_dicts:

            for pairs in shape['points']:
                points.append(ia.Keypoint(x=pairs[0], y=pairs[1]))

            _d = {}
            _d['label'] = shape['label']
            _d['index'] = (i, i+len(shape['points']))
            aug_shape_dicts.append(_d)

            i += len(shape['points'])

        keypoints = ia.KeypointsOnImage(points, shape=(256,256,3))

        seq_det = seq.to_deterministic()
        image_aug = seq_det.augment_images([im])[0]
        keypoints_aug = seq_det.augment_keypoints([keypoints])[0]

        for shape in aug_shape_dicts:
            start, end = shape['index']
            aug_points = [[keypoint.x, keypoint.y] for keypoint in keypoints_aug.keypoints[start:end]]
            shape['points'] = aug_points

        return image_aug, aug_shape_dicts


    def __data_generation(self, image_paths, annot_paths):

        X = np.empty((self.batch_size, imshape[0], imshape[1], imshape[2]), dtype=np.float32)
        Y = np.empty((self.batch_size, imshape[0], imshape[1], n_classes),  dtype=np.float32)

        for i, (im_path, annot_path) in enumerate(zip(image_paths, annot_paths)):

            # read image as grayscale or rgb
            if imshape[2] == 1:
                im = cv2.imread(im_path, 0)
                im = np.expand_dims(im, axis=2)
            elif imshape[2] == 3:
                im = cv2.imread(im_path, 1)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            shape_dicts = self.get_poly(annot_path)

            # check for augmentation
            if self.augment:
                im, shape_dicts = self.augment_poly(im, shape_dicts)

            # create target masks
            if n_classes == 1:
                mask = self.create_binary_masks(im, shape_dicts)
            elif n_classes > 1:
                mask = self.create_multi_masks(im, shape_dicts)

            X[i,] = im
            Y[i,] = mask

        return X, Y
