import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import os
import time
import contextlib
import numpy as np
import cv2
from utils import VideoStream
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from models import preprocess_input, dice
from config import imshape, model_name, n_classes
from utils import add_masks, crf
with contextlib.redirect_stdout(None):
    import pygame


RUN = True
MODE = 'softmax'
CALC_CRF = False
BACKGROUND = False

frame_shape = (640, 480)
target_shape = imshape[:2]
d_width = target_shape[0] // 2
d_height = target_shape[1] // 2
x0 = frame_shape[1] // 2 - d_width
y0 = frame_shape[0] // 2 - d_height
x1 = frame_shape[1] // 2 + d_width
y1 = frame_shape[0] // 2 + d_height

model = load_model(os.path.join('models', model_name+'.model'),
                   custom_objects={'dice': dice})

pygame.init()
screen = pygame.display.set_mode(frame_shape)
vs = VideoStream(device=0).start()
time.sleep(0.1)
prev = time.time()

while RUN:

    current = time.time()

    # camera stream
    if vs.check_queue():

        delta = current - prev
        prev = current

        screen.fill([0,0,0])
        frame = vs.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = frame.copy()

        roi = im[x0:x1, y0:y1]
        tmp = np.expand_dims(roi, axis=0)
        roi_pred = model.predict(tmp)

        if MODE == 'argmax':
            if n_classes == 1:
                roi_pred = roi_pred.squeeze()
                roi_softmax = np.stack([1-roi_pred, roi_pred], axis=2)
                roi_max = np.argmax(roi_softmax, axis=2)
                roi_pred = np.array(roi_max, dtype=np.float32)
            elif n_classes > 1:
                roi_max = np.argmax(roi_pred.squeeze(), axis=2)
                roi_pred = to_categorical(roi_max)

        if CALC_CRF:
            if n_classes == 1:
                roi_pred = roi_pred.squeeze()
                roi_softmax = np.stack([1-roi_pred, roi_pred], axis=2)
                roi_mask = crf(roi_softmax, roi)
                roi_mask = np.array(roi_mask, dtype=np.float32)
                roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2RGB)
            elif n_classes > 1:
                roi_mask = crf(roi_pred.squeeze(), roi)

        else:
            if n_classes == 1:
                roi_mask = roi_pred.squeeze()*255.0
                roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2RGB)
            elif n_classes > 1:
                roi_mask = add_masks(roi_pred.squeeze()*255.0)

        if BACKGROUND:
            roi_mask = np.array(roi_mask, dtype=np.uint8)
            roi_mask = cv2.addWeighted(roi, 1.0, roi_mask, 1.0, 0)

        frame[x0:x1, y0:y1] = roi_mask
        cv2.rectangle(frame, (y0, x0), (y1, x1), (0,0,255), 2)
        cv2.putText(frame, 'FPS: '+str(np.round(1/delta, 1)), (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'MODE: '+str(MODE), (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'CRF: '+str(CALC_CRF), (10,110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
        frame = cv2.flip(frame, 0)
        frame = np.rot90(frame, k=3)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0,0))
        pygame.display.update()

        for event in pygame.event.get():

            keys = pygame.key.get_pressed()

            # Close window by pressing X in top right
            if event.type == pygame.QUIT:
                RUN = False

            # Press Q to close window
            elif event.type == pygame.KEYDOWN and keys[pygame.K_q]:
                RUN = False

            # C turns on Conditional Random Field Processing
            elif event.type == pygame.KEYDOWN and keys[pygame.K_c]:
                CALC_CRF = not(CALC_CRF)

            # M switches from probability to argmax output
            elif event.type == pygame.KEYDOWN and keys[pygame.K_m]:
                if MODE == 'softmax':
                    MODE = 'argmax'
                else:
                    MODE = 'softmax'

            # B toggles background on/ off
            elif event.type == pygame.KEYDOWN and keys[pygame.K_b]:
                if BACKGROUND == False:
                    BACKGROUND = True
                else:
                    BACKGROUND = False

cv2.destroyAllWindows()
vs.stop()
