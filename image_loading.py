from libtiff import TIFF
from matplotlib import pyplot as plt
import numpy as np
import math
import sys

def load_image(image_path):
    tif = TIFF.open(image_path, mode='r')
    print(f"Image: {image_path}")
    print("TIFF Tags:")
    for tag in ['ImageWidth', 'ImageLength', 'BitsPerSample', 'Compression',
                'Photometric', 'SamplesPerPixel', 'RowsPerStrip', 'StripByteCounts',
                'PlanarConfig', 'TileWidth', 'TileLength', 'TileDepth']:
        value = tif.GetField(tag)
        if value is not None:
            print(f"  {tag}: {value}")
        else:
            print(f"  {tag}: not set")
    is_tiled = tif.IsTiled()
    print(f"Is Tiled: {is_tiled}")
    try:
        width = tif.GetField('ImageWidth')
        height = tif.GetField('ImageLength')
        print(f"width: {width}, height: {height}")
    except Exception as e:
        print(f"could not get image dimensions: {e}")
        sys.exit(1)

    samples_per_pixel = tif.GetField('SamplesPerPixel') or 1
    print(f"Samples Per Pixel: {samples_per_pixel}")

    # preparing the image array
    if samples_per_pixel == 1:
        image = np.zeros((height, width), dtype=np.uint8)
    else:
        image = np.zeros((height, width, samples_per_pixel), dtype=np.uint8)

    if is_tiled:
        tile_width = tif.GetField('TileWidth')
        tile_length = tif.GetField('TileLength')
        print(f"Tile Width: {tile_width}, Tile Length: {tile_length}")
        tiles_x = int(math.ceil(width / tile_width))
        tiles_y = int(math.ceil(height / tile_length))
        print(f"Tiles in X: {tiles_x}, Tiles in Y: {tiles_y}")
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                x = tx * tile_width
                y = ty * tile_length
                try:
                    tile = tif.read_one_tile(x, y)
                except Exception as e:
                    print(f"Could not read tile at ({x}, {y}): {e}")
                    continue
                x_end = min(x + tile_width, width)
                y_end = min(y + tile_length, height)
                tile_width_actual = x_end - x
                tile_length_actual = y_end - y
                if samples_per_pixel == 1:
                    image[y:y_end, x:x_end] = tile[0:tile_length_actual, 0:tile_width_actual]
                else:
                    if len(tile.shape) == 3:
                        image[y:y_end, x:x_end, :] = tile[0:tile_length_actual, 0:tile_width_actual, :]
                    else:
                        print(f"Unexpected tile shape at ({x}, {y}): {tile.shape}")
                        continue
    else:
        # for stripped images, we want to just read the whole image
        try:
            full_image = tif.read_image()
            print("Image read using read_image()")
            if samples_per_pixel == 1:
                image = full_image
            else:
                image = full_image.reshape((height, width, samples_per_pixel))
        except Exception as e:
            print(f"Could not read the image: {e}")
            sys.exit(1)
    tif.close()
    if samples_per_pixel == 1:
        plt.imshow(image, cmap='gray')
    else:
        if samples_per_pixel >= 3:
            plt.imshow(image[:, :, :3])
        else:
            plt.imshow(image)
    plt.title(image_path)
    plt.show()

def load_svs_subimages(svs_file):
    tif = TIFF.open(svs_file, mode='r')
    subimages = []
    index = 0
    for image in tif.iter_images():
        print(f"\nreading sub-image {index}")
        try:
            img = image
            height, width = img.shape[:2]
            samples_per_pixel = img.shape[2] if img.ndim == 3 else 1
            print(f"dimensions: Width={width}, Height={height}, SamplesPerPixel={samples_per_pixel}")
            subimages.append(img)
            plt.imshow(img)
            plt.title(f"sub-image {index}")
            plt.show()
            index += 1
        except Exception as e:
            print(f"error reading sub-image {index}: {e}")
            continue

    tif.close()

    return subimages

if __name__ == "__main__":
    image_files = [
        'Kidney1.tif',
        'Region_001_FOV_00041_Acridine_Or_Gray.tif',
        'Region_001_FOV_00041_DAPI_Gray.tif',
        'Region_001_FOV_00041_FITC_Gray.tif',
        'SkinOverview.tif',
        'TMA2-v2.tif'
    ]

    for image_file in image_files:
        print("\nProcessing:", image_file)
        try:
            load_image(f"imageset1/{image_file}")
        except Exception as e:
            print(f"An error occurred while processing {image_file}: {e}")
    #part 2
    svs_file = 'Kidney2_RGB2_20x.svs'
    try:
        load_svs_subimages(svs_file)
    except Exception as e:
        print(f"An error occurred while processing the SVS file {svs_file}: {e}")
