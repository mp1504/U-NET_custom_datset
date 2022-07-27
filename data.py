
import numpy as np
import glob, os, argparse
from keras.utils import img_to_array, load_img

def argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description = 'U-NET implementation using Keras')
    parser.add_argument('--height', type=int, default=256, help='height of the resized image', metavar='')
    parser.add_argument('--width', type=int, default=256, help='height of the resized image', metavar='')
    return parser.parse_args()

class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path="./data/train/images", label_path="./data/train/labels",
                 test_path="./data/test/images", npy_path="./data/npydata", img_type="png"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self):
        print('\nCreating training images...')
        imgs = glob.glob(self.data_path+os.sep+"*."+self.img_type)
        print('Total images :', len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)

        for x in range(len(imgs)):
            imgpath = imgs[x]
            pic_name = imgpath.split(os.sep)[-1]
            labelpath = self.label_path + os.sep + pic_name
            img = load_img(imgpath, color_mode="rgb", target_size=[self.out_rows, self.out_cols])
            label = load_img(labelpath, color_mode = "grayscale", target_size=[self.out_rows, self.out_cols])
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[x] = img
            imglabels[x] = label
            if x % 100 == 0 and x > 0:
                print('Done :', x,'images')

        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('\nCreating test images...')
        imgs = glob.glob(self.test_path + os.sep + "*." + self.img_type)
        print('Total images :', len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        testpathlist = []

        for imgname in imgs:
            testpath = imgname
            testpathlist.append(testpath)
            img = load_img(testpath, color_mode="rgb", target_size=[self.out_rows, self.out_cols])
            img = img_to_array(img)
            imgdatas[i] = img
            if i % 100 == 0 and i > 0:
                print('Done :', i,'images')
            i += 1

        txtname = './data/results/pic.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('load train images...')
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1  
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test

if __name__ == "__main__":
    args = argparser()
    mydata = dataProcess(args.height, args.width)
    mydata.create_train_data()
    mydata.create_test_data()
