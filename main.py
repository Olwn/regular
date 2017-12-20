import argparse
from datetime import datetime
import keras
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.optimizers import SGD
from keras.datasets import cifar10
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

lr_scheduler = LearningRateScheduler(lambda x: 0.05 if x < 60 else 0.03 if x < 80 else 0.005)
dir_name = "e" + datetime.now().strftime("%m%d-%H-%M-%S") + \
           "layers" + str(params.ly) + \
           "_ing"

board = TensorBoard(log_dir=dir_name, histogram_freq=0, write_graph=False, write_images=False)
model.fit(x_train, y_train, batch_size=params.bs, epochs=params.ep)
score = model.evaluate(x_test, y_test, batch_size=32)
