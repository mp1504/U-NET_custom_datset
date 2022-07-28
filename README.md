# A simple U-Net implementation for custom datasets

To run the U-Net:

1. Create data folder where unet.py is present and create subfolders './data/results', './data/train' and './data/test'

2. Place your training images and their labels(mask) inside './data/train/images' and './data/train/labels' respectively and place your testing images under './data/test/images'. All images must be of type png
   
3. Execute 'unet.py' with 'train' operation to train your dataset. You can check available script options using 'python unet.py -h'. Example-
   
   ```bash
   python unet.py --operation train --batch_size 16 --epochs 50 --image_size 512
   ```

4. Once training is complete, you can get results by executing 'unet.py' in 'test' operation. Results will be placed under './data/results'
   
   ```bash
   python unet.py --operation test --batch_size 16 --image_size 512
   ```

5. Trained weights can be found in the checkpoint directory - 'unet.hdf5'