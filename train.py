import os
from tensorflow.keras.callbacks import ModelCheckpoint
from data_generator import DataGenerator
from tensorboard_callbacks import TrainValTensorBoard, TensorBoardMask
from utils import generate_missing_json
from config import model_name, n_classes
from models import unet, fcn_8


def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0]))

if len(os.listdir('images')) != len(os.listdir('annotated')):
    generate_missing_json()

image_paths = [os.path.join('images', x) for x in sorted_fns('images')]
annot_paths = [os.path.join('annotated', x) for x in sorted_fns('annotated')]

if 'unet' in model_name:
    model = unet(pretrained=False, base=4)
elif 'fcn_8' in model_name:
    model = fcn_8(pretrained=False, base=4)

tg = DataGenerator(image_paths=image_paths, annot_paths=annot_paths,
                   batch_size=5, augment=True)

checkpoint = ModelCheckpoint(os.path.join('models', model_name+'.model'), monitor='dice', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=10)

train_val = TrainValTensorBoard(write_graph=True)
tb_mask = TensorBoardMask(log_freq=10)

model.fit_generator(generator=tg,
                    steps_per_epoch=len(tg),
                    epochs=500, verbose=1,
                    callbacks=[checkpoint, train_val, tb_mask])
