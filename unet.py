# -*- coding:utf-8 -*-
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import array_to_img, ImageDataGenerator
from random import randint

import tensorflow as tf
import numpy as np
import cv2, argparse
import os, glob, math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(1)

def argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description = 'U-NET implementation using Keras')
    parser.add_argument('--operation', type=str, required=True, help='whether \'train\' or \'test\'', metavar='')
    parser.add_argument('--checkpoint_dir', type=str, default='.', help='where to save trained model, parameters and logs', metavar='')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training the model', metavar='')
    parser.add_argument('--image_size', type=int, default=256, help='image will be resized to (size X size) pixels', metavar='')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs for training the model', metavar='')
    parser.add_argument('--initial_epoch', type=int, default=1, help='starting epoch for continuing model training', metavar='')
    parser.add_argument('--validation_split', type=float, default=0.1, help='fraction of training data to be used for validation', metavar='')
    parser.add_argument('--shuffle', action='store_true', help='whether to shuffle the training data before each epoch')
    parser.add_argument('--continue_training', action='store_true', help='continue training the previous model')
    parser.add_argument('--verbose', type=int, default=1, help='verbose level : 0 = silent, 1 = progress bar, 2 = one line per epoch', metavar='')
    return parser.parse_args()

def print_params(args):
    message = ''
    message += '---------------- Options ----------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>19} : {:<7}{}\n'.format(str(k), str(v), comment)
    message += '------------------ End ------------------\n'
    print(message)
    if args.operation == 'train':
        with open(args.checkpoint_dir + '/train_params.txt', 'wt') as opt_file:
            opt_file.write(message)

class ImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, validation_generator):
        self.img_dir = checkpoint_dir + '/images/'
        self.generator = validation_generator
        try: os.mkdir(self.img_dir)
        except OSError: pass

    def on_epoch_end(self, epoch, logs={}):
        validation_data = next(self.generator)
        idx = randint(1, len(validation_data[0])) - 1
        x_test = validation_data[0][idx]
        y_test = validation_data[1][idx]
        y_pred = self.model.predict(np.reshape(x_test, (1,) + x_test.shape), verbose=0)[0]
        array_to_img(x_test).save(self.img_dir + '/epoch{0:03d}_real_A.png'.format(epoch+1))
        array_to_img(y_test).save(self.img_dir + '/epoch{0:03d}_real_B.png'.format(epoch+1))
        array_to_img(y_pred).save(self.img_dir + '/epoch{0:03d}_fake_B.png'.format(epoch+1))

class myUnet(object):
    def __init__(self, _image_size):
        self.imgage_size = _image_size
        self.get_unet()

    def get_unet(self):
        inputs = Input((self.imgage_size, self.imgage_size, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        up6 = BatchNormalization()(up6)
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        up7 = BatchNormalization()(up7)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        up8 = BatchNormalization()(up8)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        up9 = BatchNormalization()(up9)
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model

    def train(self, checkpoint_dir, continue_training, _batch_size, _epochs, initial_epoch, _verbose, _validation_split, _shuffle):
        if continue_training:
            self.model.load_weights(checkpoint_dir + '/unet.hdf5')
        elif os.path.exists(checkpoint_dir + '/unet.hdf5'):
            os.remove(checkpoint_dir + '/unet.hdf5')
        
        image_generator = ImageDataGenerator(rescale = (1/255), validation_split = _validation_split)
        _seed = randint(100,100000)
        train_image_generator = image_generator.flow_from_directory(
            './data/train/',
            classes = ['images'],
            class_mode=None,
            target_size = (self.imgage_size, self.imgage_size), 
            color_mode = 'rgb', 
            batch_size = _batch_size, 
            shuffle = _shuffle,
            seed = _seed,
            subset = 'training',
        )
        train_label_generator = image_generator.flow_from_directory(
            './data/train/',
            classes = ['labels'],
            class_mode=None,
            target_size = (self.imgage_size, self.imgage_size), 
            color_mode = 'grayscale', 
            batch_size = _batch_size, 
            shuffle = _shuffle,
            seed = _seed,
            subset = 'training',
        )
        _seed = randint(100,100000)
        validation_image_generator = image_generator.flow_from_directory(
            './data/train/',
            classes = ['images'],
            class_mode=None,
            target_size = (self.imgage_size, self.imgage_size), 
            color_mode = 'rgb', 
            batch_size = _batch_size, 
            shuffle = _shuffle,
            seed = _seed,
            subset = 'validation',
        )
        validation_label_generator = image_generator.flow_from_directory(
            './data/train/',
            classes = ['labels'],
            class_mode=None,
            target_size = (self.imgage_size, self.imgage_size), 
            color_mode = 'grayscale', 
            batch_size = _batch_size, 
            shuffle = _shuffle,
            seed = _seed,
            subset = 'validation',
        )
        train_generator = zip(train_image_generator, train_label_generator)
        validation_generator = zip(validation_image_generator, validation_label_generator)
        
        model_checkpoint = ModelCheckpoint(checkpoint_dir + '/unet.hdf5', monitor='loss', verbose=0, save_best_only=True)
        tensorboard_callback = TensorBoard(log_dir=checkpoint_dir+'/tensorboard_logs/', histogram_freq=1)
        
        print('Fitting model...')
        self.model.fit(train_generator, 
            epochs=_epochs + (initial_epoch-1), 
            verbose=_verbose, 
            steps_per_epoch=train_image_generator.samples // _batch_size,
            validation_data=validation_generator,
            validation_steps=validation_image_generator.samples // _batch_size,
            callbacks=[model_checkpoint, tensorboard_callback, ImageCallback(checkpoint_dir, validation_generator)],
            initial_epoch=(initial_epoch-1),
        )
        self.model.save_weights(checkpoint_dir + '/unet.hdf5')
        
    def test(self, checkpoint_dir, _batch_size):
        test_generator = ImageDataGenerator(rescale = (1/255)).flow_from_directory(
            './data/test/',
            classes = ['images'],
            target_size = (self.imgage_size, self.imgage_size), 
            color_mode = 'rgb',
            shuffle=False,
            batch_size = _batch_size,
        )
        print('predict test data')
        self.model.load_weights(checkpoint_dir + '/unet.hdf5')
        num_images = len(glob.glob('./data/test/images/*.png'))
        _steps = math.ceil(num_images / _batch_size)
        imgs_mask_test = self.model.predict(test_generator, steps=_steps, verbose=1)
        
        print("Saving images")
        piclist = [i.split(os.sep)[-1] for i in glob.glob('./data/test/images/*.png')]
        test_img_shape = cv2.imread(glob.glob('./data/test/images'+os.sep+'*.png')[0]).shape
        height, width = test_img_shape[0], test_img_shape[1]
        for i in range(imgs_mask_test.shape[0]):
            path = "./data/results/" + piclist[i]
            array_to_img(imgs_mask_test[i]).resize((width,height)).save(path)

if __name__ == '__main__':
    args = argparser()
    print_params(args)
    myunet = myUnet(args.image_size)
    if args.operation == 'train':
        myunet.train(args.checkpoint_dir, args.continue_training, args.batch_size, args.epochs, args.initial_epoch, args.verbose, args.validation_split, args.shuffle)
    elif args.operation == 'test':
        myunet.test(args.checkpoint_dir, args.batch_size)
    else:
        print('ERROR : INVALID OPERATION')
