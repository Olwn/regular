import argparse
import os
from datetime import datetime
import keras
from keras.callbacks import LearningRateScheduler, TensorBoard, CSVLogger, ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

from resnet import ResnetBuilder


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

parser = argparse.ArgumentParser()
parser.add_argument("-ly", type=int, default=14)
parser.add_argument("-wd", type=float, default=1e-4)
parser.add_argument("-bn", type=str2bool, default=False)
parser.add_argument("-ep", type=int, default=100)
parser.add_argument("-bs", type=int, default=128)
params = parser.parse_args()

model = ResnetBuilder.build_resnet_cifar10(params.ly, use_bn=params.bn)

sgd = SGD(momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


dir_name = "e" + datetime.now().strftime("%m%d-%H-%M-%S") + \
           "layers" + str(params.ly) + \
           "_ing"
csv_logger = CSVLogger(os.path.join(dir_name, 'log.csv'))
lr_scheduler = LearningRateScheduler(lambda x: 0.05 if x < 60 else 0.03 if x < 80 else 0.005)
board = TensorBoard(log_dir=dir_name, histogram_freq=0, write_graph=False, write_images=False)
checker = ModelCheckpoint(filepath=os.path.join(dir_name, "weights.{epoch:02d}-{val_loss:.2f}.hdf5"), period=20)
if not os.path.exists(dir_name): os.mkdir(dir_name)
with open(os.path.join(dir_name, 'config'), 'w') as wf:
    wf.write("-----" + str(params) + dir_name)
callbacks = [lr_scheduler, csv_logger, board, checker]
datagen_train = ImageDataGenerator(
    width_shift_range=params.shift,
    height_shift_range=params.shift,
)
datagen_test = ImageDataGenerator()
model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=params.bs),
                    steps_per_epoch=x_train.shape[0] / params.bs,
                    validation_data=datagen_test.flow(x_test, y_test, batch_size=params.bs),
                    validation_steps=x_test.shape[0] / params.bs,
                    epochs=params.ep,
                    max_q_size=2500,
                    workers=10,
                    callbacks=callbacks)
losses = model.evaluate(x_test, y_test, batch_size=params.batch_size)
print(losses)
with open(os.path.join(dir_name, 'config'), 'a') as wf:
    wf.write(str(losses) + '\n')
os.rename(dir_name, dir_name.replace('_ing', '_completed'))