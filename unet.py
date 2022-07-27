# -*- coding:utf-8 -*-

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.utils import array_to_img
from keras.callbacks import CSVLogger

import cv2, argparse
from data import *

def argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description = 'U-NET implementation using Keras')
    parser.add_argument('--operation', type=str, default='train', help='whether \'train\' or \'test\'', metavar='')
    parser.add_argument('--checkpoint_dir', type=str, default='.', help='where to save trained model, parameters and logs', metavar='')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training the model', metavar='')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training the model', metavar='')
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

class myUnet(object):
    def __init__(self):
        train_shape = np.load('./data/npydata/imgs_test.npy').shape
        self.img_rows = train_shape[1]
        self.img_cols = train_shape[2]
        self.get_unet()

    def load_train_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_test = mydata.load_test_data()
        return imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print(conv1)
        conv1 = BatchNormalization()(conv1)
        print ("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        print ("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print ("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print ("conv2 shape:", conv2.shape)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print ("conv2 shape:", conv2.shape)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print ("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print ("conv3 shape:", conv3.shape)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print ("conv3 shape:", conv3.shape)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print ("pool3 shape:", pool3.shape)

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
        print(up6)
        print(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        print(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        print(conv6)
        conv6 = BatchNormalization()(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        up7 = BatchNormalization()(up7)
        merge7 = concatenate([conv3, up7], axis=3)
        print(up7)
        print(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        print(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        print(conv7)
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
        print(up9)
        print(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        print(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        print(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)
        print ("conv9 shape:", conv9.shape)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        print(conv10)
        model = Model(inputs=inputs, outputs=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model

    def train(self, checkpoint_dir, continue_training, _batch_size, _epochs, _verbose, _validation_split, _shuffle):
        print("loading data")
        imgs_train, imgs_mask_train = self.load_train_data()
        print("loading data done")
        self.load_model_weights(checkpoint_dir, continue_training)
        model_checkpoint = ModelCheckpoint(checkpoint_dir + '/unet.hdf5', monitor='val_loss', verbose=0, save_best_only=True)
        csv_logger = CSVLogger(checkpoint_dir + '/model_history_log.csv', append=continue_training)
        print('Fitting model...')
        self.model.fit(x=imgs_train, y=imgs_mask_train, batch_size=_batch_size, epochs=_epochs, verbose=_verbose,
                  validation_split=_validation_split, shuffle=_shuffle , callbacks=[model_checkpoint, csv_logger])
        self.model.save_weights(checkpoint_dir + '/unet.hdf5')
        
    def test(self, checkpoint_dir):
        print("loading data")
        imgs_test = self.load_test_data()
        print("loading data done")
        print('predict test data')
        self.model.load_weights(checkpoint_dir + '/unet.hdf5')
        imgs_mask_test = self.model.predict(imgs_test, batch_size=1, verbose=1)
        print("Saving images")
        np.save('./data/results/imgs_mask_test.npy', imgs_mask_test)
        piclist = []
        for line in open("./data/results/pic.txt"):
            line = line.strip()
            picname = line.split(os.sep)[-1]
            piclist.append(picname)
        print(len(piclist))
        for i in range(imgs_mask_test.shape[0]):
            path = "./data/results/" + piclist[i]
            img = imgs_mask_test[i]
            img = array_to_img(img)
            img.save(path)
            train_img = cv2.imread(glob.glob('./data/train/labels'+os.sep+'*.png')[0], cv2.IMREAD_GRAYSCALE)
            height, width = train_img.shape
            cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            cv_pic = cv2.resize(cv_pic,(width,height))
            binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(path, cv_save)
            
    def load_model_weights(self, checkpoint_dir, continue_training):
        weight_file = checkpoint_dir + '/unet.hdf5'
        if continue_training:
            self.model.load_weights(weight_file)
        else:
            if os.path.exists(weight_file):
                os.remove(weight_file)

if __name__ == '__main__':
    args = argparser()
    print_params(args)
    myunet = myUnet()
    if args.operation == 'train':
        myunet.train(args.checkpoint_dir, args.continue_training, args.batch_size, args.epochs, args.verbose, args.validation_split, args.shuffle)
    elif args.operation == 'test':
        myunet.test(args.checkpoint_dir)
    else:
        print('ERROR : INVALID OPERATION')
