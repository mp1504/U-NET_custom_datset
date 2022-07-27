# A simple U-Net implementation for custom datasets

To run the U-Net:
1. Create a folder data in the same directory as other files
2. Create folders npydata, results, train and test inside data folder
3. Create folders images and labels inside both the train and test folders. All images must of type png (modify data.py for using other image types)
4. Place your training images and their labels(mask) inside './data/train/images' and './data/train/labels' and place your testing images under './data/test/images'
5. Execute data.py to process dataset images and labels
6. Execute unet.py and wait for the training to happen. You can check available script options using 'python unet.py -h'
7. Trained weights can be found in the root directory - 'unet.hdf5'
8. Once complete, your results will be placed under './data/results'
