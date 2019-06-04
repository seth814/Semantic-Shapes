import cv2
from queue import Queue
from threading import Thread
import os
from config import imshape
import json
import numpy as np
from config import hues, labels, imshape, mode
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from tensorflow.keras.utils import to_categorical


class VideoStream:
    def __init__(self, device=0, size=100):
        self.stream = cv2.VideoCapture(device)
        self.stream.set(cv2.CAP_PROP_FPS, 10)
        self.stopped = False
        self.queue = Queue(maxsize=size)

    def start(self):
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        return self

    def update(self):
        while self.stopped is False:

            if not self.queue.full():
                (grabbed, frame) = self.stream.read()

            if not grabbed:
                self.stop()
                return

            self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def check_queue(self):
        return self.queue.qsize() > 0

    def stop(self):
        self.stopped = True
        self.stream.release()


def generate_missing_json():

    # creates a background json for the entire image if missing
    # this assumes you will never annotate a background class

    for im in os.listdir('images'):
        fn = im.split('.')[0]+'.json'
        path = os.path.join('annotated', fn)

        if os.path.exists(path) is False:
            json_dict = {}

            # these points might be reversed if not using a square image (idk)
            json_dict['shapes'] = [{"label": "background",
                                    "points": [[0,0],
                                               [0, imshape[0]-1],
                                               [imshape[0]-1, imshape[1]-1],
                                               [imshape[0]-1, 0]]
                                    }]
            with open(path, 'w') as handle:
                json.dump(json_dict, handle, indent=2)


def add_masks(pred):
    blank = np.zeros(shape=imshape, dtype=np.uint8)

    for i, label in enumerate(labels):

        hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
        sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
        val = pred[:,:,i].astype(np.uint8)

        im_hsv = cv2.merge([hue, sat, val])
        im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        blank = cv2.add(blank, im_rgb)

    return blank


def crf(im_softmax, im_rgb):
    n_classes = im_softmax.shape[2]
    feat_first = im_softmax.transpose((2, 0, 1)).reshape(n_classes, -1)
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)
    im_rgb = np.ascontiguousarray(im_rgb)

    d = dcrf.DenseCRF2D(im_rgb.shape[1], im_rgb.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(5, 5), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(5, 5), srgb=(13, 13, 13), rgbim=im_rgb,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((im_rgb.shape[0], im_rgb.shape[1]))
    if mode is 'binary':
        return res * 255.0
    if mode is 'multi':
        res_hot = to_categorical(res) * 255.0
        res_crf = add_masks(res_hot)
        return res_crf
