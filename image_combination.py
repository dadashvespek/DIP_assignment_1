from libtiff import TIFF
from matplotlib import pyplot as plt
import numpy as np

def load_image(filename):
    try:
        tif = TIFF.open(filename, mode='r')
        image = tif.read_image()
        return image
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return None

def combine_fluorescence_images(red_file, green_file, blue_file):
    red_image = load_image(red_file)
    green_image = load_image(green_file)
    blue_image = load_image(blue_file)

    if red_image is None or green_image is None or blue_image is None:
        print("Error: One or more images failed to load.")
        return None
    if not (red_image.shape == green_image.shape == blue_image.shape):
        print("Error: Images have different shapes.")
        return None

    rgb_image = np.stack((red_image, green_image, blue_image), axis=-1)
    # normalizing
    rgb_image = (rgb_image / rgb_image.max() * 255).astype(np.uint8)

    return rgb_image

def save_image(image, filename):
    tif = TIFF.open(filename, mode='w')
    tif.write_image(image)
    tif.close()

red_file = 'imageset1/Region_001_FOV_00041_Acridine_Or_Gray.tif'
green_file = 'imageset1/Region_001_FOV_00041_FITC_Gray.tif'
blue_file = 'imageset1/Region_001_FOV_00041_DAPI_Gray.tif'

combined_image = combine_fluorescence_images(red_file, green_file, blue_file)

if combined_image is not None:
    save_image(combined_image, 'combined_fluorescence.tif')
    print("Combined image saved as 'combined_fluorescence.tif'")
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_image, interpolation='nearest', vmin=0, vmax=255)
    plt.title('Combined Fluorescence Image')
    plt.axis('off')
    plt.show()