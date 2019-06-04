import cv2
import os
from skimage.io import imsave
import time
import contextlib
import numpy as np
from utils import VideoStream
from config import imshape
with contextlib.redirect_stdout(None):
    import pygame


RUN = True
frame_shape = (640, 480)
target_shape = imshape[:2]
save_dir = 'images'

d_width = target_shape[0] // 2
d_height = target_shape[1] // 2
x0 = frame_shape[1] // 2 - d_width
y0 = frame_shape[0] // 2 - d_height
x1 = frame_shape[1] // 2 + d_width
y1 = frame_shape[0] // 2 + d_height

if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
images = sorted(os.listdir(save_dir), key=lambda x: int(x.split('.')[0]))

if images != []:
    tmp = images[-1]
    i = int(tmp.split('.')[0]) + 1
else:
    i = 0

pygame.init()
screen = pygame.display.set_mode(frame_shape)
vs = VideoStream(device=0).start()
time.sleep(0.1)
prev = time.time()

while RUN:

    current = time.time()

    # update camera display
    if vs.check_queue():

        screen.fill([0,0,0])
        frame = vs.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = frame.copy()
        cv2.rectangle(frame, (y0, x0), (y1, x1), (0,255,0), 2)
        frame = np.rot90(frame, k=1)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0,0))
        pygame.display.update()

    # query keyboard 100 Hz
    if current - prev > 1/100:

        for event in pygame.event.get():

            keys = pygame.key.get_pressed()

            if event.type == pygame.QUIT:
                RUN = False

            elif event.type == pygame.KEYDOWN and keys[pygame.K_s]:
                roi = im[x0:x1, y0:y1]
                imsave(os.path.join(save_dir, '{}.jpg'.format(i)), roi)
                print('Saved image: {}.jpg'.format(i))
                i += 1

            elif event.type == pygame.KEYDOWN and keys[pygame.K_q]:
                RUN = False


        delta = current - prev
        prev = current

cv2.destroyAllWindows()
vs.stop()
