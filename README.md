# A simple U-Net implementation for custom datasets

To run the U-Net:

1. Create a folder data in the same directory as other files

2. Create folders npydata, results, train and test inside data folder

3. Create folders images and labels inside both the train and test folders. All images must of type png

4. Place your training images and their labels(mask) inside './data/train/images' and './data/train/labels' and place your testing images under './data/test/images'

5. Execute 'data.py' to process dataset images and labels. You can check available script options using 'python data.py -h'. Example-
   
   ```bash
   python data.py --height 256 --width 256
   ```

6. Execute 'unet.py' with 'train' operation to train your dataset. You can check available script options using 'python unet.py -h'. Example-
   
   ```bash
   python unet.py --operation train --batch_size 16 --epochs 50
   ```

7. Once training is complete, you can get results by executing 'unet.py' in 'test' operation. Results will be placed under './data/results'
   
   ```bash
   python unet.py --operation test
   ```

8. Trained weights can be found in the checkpoint directory - 'unet.hdf5'