from PIL import Image
import io
import tensorflow as tf
import os
import cv2
import numpy as np
from skimage.io import imsave
from config import model_name, logbase, imshape, labels, hues, n_classes
import shutil


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        tmp = os.path.join(logbase, 'metrics')
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
            os.mkdir(tmp)
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(logbase, 'metrics', model_name+'_train')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        val_log_dir = os.path.join(logbase, 'metrics', model_name+'_val')
        self.val_log_dir = val_log_dir

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)


class TensorBoardMask(tf.keras.callbacks.Callback):
    def __init__(self, log_freq):
        super().__init__()
        self.log_freq = log_freq
        self.im_summaries = []
        self.global_batch = 0
        tmp = os.path.join(logbase, 'images')
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
            os.mkdir(tmp)
        self.logdir = tmp
        self.writer = tf.summary.FileWriter(self.logdir)
        self.write_summaries()


    def _file_generator(self, path):
        files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        for fn in files:
            yield fn


    def make_image(self, path):
        """
        Convert an numpy representation image to Image protobuf.
        Modified from: https://github.com/lanpa/tensorboard-pytorch
        Colormap formating: https://stackoverflow.com/questions/10965417/how-to-\
            convert-numpy-array-to-pil-image-applying-matplotlib-colormap
        """

        image = Image.open(path)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        summary = tf.Summary.Image(height=imshape[0],
                                   width=imshape[1],
                                   colorspace=imshape[2],
                                   encoded_image_string=image_string)
        return summary


    def log_mask(self):
        for i, fn in enumerate(self._file_generator(logbase)):
            mask = self.predict(os.path.join(logbase, fn))
            save_path = os.path.join(self.logdir, 'mask_{}.png'.format(i))
            imsave(save_path, mask)
            image_summary = self.make_image(save_path)
            self.im_summaries.append(tf.Summary.Value(tag='mask_{}'.format(i), image=image_summary))


    def add_masks(self, pred):
        blank = np.zeros(shape=imshape, dtype=np.uint8)

        for i, label in enumerate(labels):

            hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
            sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
            val = pred[:,:,i].astype(np.uint8)

            im_hsv = cv2.merge([hue, sat, val])
            im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
            blank = cv2.add(blank, im_rgb)

        return blank


    def predict(self, path):
        if imshape[2] == 1:
            im = cv2.imread(path, 0)
            im = im.reshape(im.shape[0], imshape[1], 1)
        elif imshape[2] == 3:
            im = cv2.imread(path, 1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.reshape(im.shape[0], im.shape[1], 3)
        im = np.expand_dims(im, axis=0)
        pred = self.model.predict(im)
        pred = np.squeeze(pred) * 255.0
        if n_classes == 1:
            mask = np.array(pred, dtype=np.uint8)
        elif n_classes > 1:
            mask = self.add_masks(pred)
        return mask


    def write_summaries(self):
        summary = tf.Summary(value=self.im_summaries)
        self.writer.add_summary(summary, self.global_batch)
        self.im_summaries = []


    def on_epoch_end(self, epoch, logs={}):
        # returns if not multiple of log freq
        if int(epoch % self.log_freq) != 0:
            return

        self.log_mask()
        self.write_summaries()
        self.global_batch += self.log_freq
