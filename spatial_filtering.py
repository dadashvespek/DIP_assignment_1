from libtiff import TIFF
from matplotlib import pyplot as plt
import numpy as np
import math
import sys
import os
from image_processor import ImageProcessor

def process_image(file_path, process_func):
    tif = TIFF.open(file_path, mode='r')
    image = tif.read_image()
    tif.close()

    processor = ImageProcessor(width=image.shape[1], height=image.shape[0])
    processor.load_8bit_image(image)
    processor.convert_8bit_to_float()
    processed_float = process_func(processor.get_float_image())
    processed_processor = ImageProcessor(width=processed_float.shape[1], height=processed_float.shape[0])
    processed_processor.image_float = processed_float
    processed_processor.convert_float_to_8bit()

    return image, processed_processor.get_8bit_image()

def blur_image_averaging(image, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='edge')
    blurred = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            blurred[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    return blurred

def blur_image_gaussian(image, kernel_size=5, sigma=1):
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)

    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='edge')
    blurred = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            blurred[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    return blurred

def median_filter(image, kernel_size=3):
    padded_image = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='edge')
    filtered = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered[i, j] = np.median(neighborhood)
    return filtered


def unsharp_mask(image, kernel_size=5, sigma=1, amount=1.5, threshold=0):
    blurred = blur_image_gaussian(image, kernel_size, sigma)
    mask = image - blurred
    sharpened = image + amount * mask
    return np.clip(sharpened, 0, 1)

def laplacian_filter(image):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    laplacian = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            laplacian[i, j] = np.sum(padded_image[i:i+3, j:j+3] * kernel)
    return laplacian

def laplacian_sharpening(image, amount=1):
    laplacian = laplacian_filter(image)
    sharpened = image - amount * laplacian
    return np.clip(sharpened, 0, 1)


def sobel_gradient(image):
    #sobel operators
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    gradient_x = np.zeros_like(image)
    gradient_y = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gradient_x[i, j] = np.sum(padded_image[i:i+3, j:j+3] * sobel_x)
            gradient_y[i, j] = np.sum(padded_image[i:i+3, j:j+3] * sobel_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())

    return gradient_magnitude

def main():
    image_folder = 'imageset3'
    image_files = [
        'Fig0333(a)(test_pattern_blurring_orig).tif',
        'Fig0334(a)(hubble-original).tif',
        'Fig0335(a)(ckt_board_saltpep_prob_pt05).tif',
        'Fig0338(a)(blurry_moon).tif',
        'Fig0340(a)(dipxe_text).tif',
        'Fig0342(a)(contact_lens_original).tif',
        'Fig0343(a)(skeleton_orig).tif'
    ]

    file_path = os.path.join(image_folder, image_files[2])
    original, processed_avg = process_image(file_path, lambda img: blur_image_averaging(img, kernel_size=3))
    _, processed_gaussian = process_image(file_path, lambda img: blur_image_gaussian(img, kernel_size=3, sigma=1))
    _, processed_median = process_image(file_path, lambda img: median_filter(img, kernel_size=3))


    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(processed_avg, cmap='gray')
    plt.title('Averaging Blur')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(processed_gaussian, cmap='gray')
    plt.title('Gaussian Blur')
    plt.axis('off')

    plt.subplot(144)
    plt.imshow(processed_median, cmap='gray')
    plt.title('Median Filter')
    plt.axis('off')


    plt.tight_layout()
    plt.show()

    file_path = os.path.join(image_folder, image_files[3])  

    original, processed_unsharp = process_image(file_path, lambda img: unsharp_mask(img, kernel_size=5, sigma=1, amount=1.5))
    _, processed_laplacian = process_image(file_path, lambda img: laplacian_sharpening(img, amount=0.5))

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(processed_unsharp, cmap='gray')
    plt.title('Unsharp Mask')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(processed_laplacian, cmap='gray')
    plt.title('Laplacian Sharpening')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    file_path = os.path.join(image_folder, image_files[5])  

    original, processed_sobel = process_image(file_path, sobel_gradient)

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(processed_sobel, cmap='gray')
    plt.title('Sobel Gradient Magnitude')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()