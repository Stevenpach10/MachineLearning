import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from guided_filter_pytorch.guided_filter import FastGuidedFilter
from tqdm import tqdm
import sys, getopt


def images_iterable(image_dir, batch_size=10):
    """
    Defines an iterable for the images .jpg stored it in the image_dir and return the images 
    defined by the batch_size.

    Parameters
    ----------
    image_dir : str
        The images directory where the images are stored
    batch_size : int, optional
        The batch size of return images for each iteration
    """
    batch = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            file_path = os.path.join(image_dir, filename)
            with Image.open(file_path) as image:
                batch.append((filename, image.copy()))
                if len(batch) == batch_size:
                    yield batch
                    batch = []

    # Yield the remaining batch (if any)
    if batch:
        yield batch

def gen_detailed_image(image_dir,
                    image_gen, 
                    image_detailed_dir_output = 'Hands_Detailed', 
                    image_filter_dir_output = 'Hands_Smooth', 
                    width=255, 
                    height = 255):
    """
    Generate the detailed images and smoothed images from an iterable images
    Apply the GaussianBlur filter and also divide the original image by the filtered
    to get the detailed image.

    Store both images in the image_detailed_dir_output and image_filter_dir_output directory
    If the directories doesn't exists, create it
    Parameters
    ----------
    image_dir : str
        Where the images are located
    image_gen : iterable
        An iterable for dataset images
    image_detailed_dir_output : str, optional
        The output directory for the detailed images
    image_filter_dir_output : str, optional
        The output directory for the smoothed images
    width : int, optional
        The width of the detailed and smoothed images
    height : int, optional
        The height of the detailed and smoothed images
    """
    eps = 1e-6
    # Check if the folder exists, and if not, create it
    if not os.path.exists(image_detailed_dir_output):
        os.makedirs(image_detailed_dir_output)
    if not os.path.exists(image_filter_dir_output):
        os.makedirs(image_filter_dir_output)

    progress_bar = tqdm(total=len(os.listdir(image_dir)))
    total_images = 0
    for i, batch in enumerate(image_gen):
        for j, (file, image) in enumerate(batch):
            total_images += 1
            image = np.array(image)
            filter = cv2.GaussianBlur(image, (25, 25), 15)

            glow = cv2.cvtColor(filter, cv2.COLOR_RGB2GRAY)
            high = np.clip((cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)+(eps/100)) / (glow+(eps/100))*255, 0, 255)

            path_detailed = os.path.join(image_detailed_dir_output, file)
            path_smoothed = os.path.join(image_filter_dir_output, file)
            
            cv2.imwrite(path_detailed, cv2.resize(high, (width, height)))
            cv2.imwrite(path_smoothed, cv2.resize(cv2.cvtColor(filter, cv2.COLOR_BGR2RGB), (width, height)))
            # Update the progress bar
        progress_bar.update(len(batch))
    # Close the progress bar
    progress_bar.close()

def main(argv):
    image_dir = ""
    try:
      opts, args = getopt.getopt(argv,"s:",["source="])
    except getopt.GetoptError:
      print ('preprocessing.py -src <directoryName>')
      sys.exit(2)
    for opt, arg in opts:
          print(opt)
          if opt == '-s':
              image_dir = arg
    
    image_gen = images_iterable(image_dir)
    gen_detailed_image(image_dir, image_gen)

if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)