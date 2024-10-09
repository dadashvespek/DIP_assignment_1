from libtiff import TIFF
import matplotlib.pyplot as plt
import numpy as np

def piecewise_linear_transform(image, breakpoints, segments):
    out = np.zeros_like(image, dtype=float)
    
    for i, (x0, x1) in enumerate(zip([0] + breakpoints, breakpoints + [255])):
        mask = (image >= x0) & (image <= x1)
        m, b = segments[i]
        out[mask] = m * image[mask] + b
    
    return np.clip(out, 0, 255).astype(np.uint8)

def histogram_stretch(image, low_percentile=2, high_percentile=98):
    low, high = np.percentile(image, [low_percentile, high_percentile])
    stretched = np.clip((image - low) * (255 / (high - low)), 0, 255)
    return stretched.astype(np.uint8)

def threshold(image, low_threshold=100, high_threshold=200):
    breakpoints = [low_threshold, high_threshold]
    segments = [
        (0, 0),  # y = 0 for x < low_threshold
        (1, 0),  # y = x for low_threshold <= x < high_threshold
        (0, 255)  # y = 255 for x >= high_threshold
    ]
    return piecewise_linear_transform(image, breakpoints, segments)

def histogram_equalization(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    
    # CDF
    cdf = histogram.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1] 
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    
    return image_equalized.reshape(image.shape).astype(np.uint8), cdf_normalized

def piecewise_linear_transformation_function(x, breakpoints, segments):
    y = np.zeros_like(x)
    for i, (x0, x1) in enumerate(zip([0] + breakpoints, breakpoints + [255])):
        m, b = segments[i]
        mask = (x >= x0) & (x <= x1)
        y[mask] = m * x[mask] + b
    return y

def power_law_lookup_table(gamma, c=1):
    x = np.arange(256)
    lookup_table = np.clip(c * (x / 255.0) ** gamma * 255, 0, 255).astype(np.uint8)
    return lookup_table

def piecewise_linear_lookup_table(breakpoints, segments):
    x = np.arange(256)
    lookup_table = np.zeros(256)
    for i, (x0, x1) in enumerate(zip([0] + breakpoints, breakpoints + [255])):
        m, b = segments[i]
        mask = (x >= x0) & (x <= x1)
        lookup_table[mask] = m * x[mask] + b
    lookup_table = np.clip(lookup_table, 0, 255).astype(np.uint8)
    return lookup_table

#I'm not sure if I've mis-understood the assignment or not, but there isnt any mention of which image should be picked, so I've picked a random one.
tif = TIFF.open('imageset2/Fig0316(4)(bottom_left).tif', mode='r')
img = tif.read_image()

# list of our gamma values
gamma_values = [0.4, 0.6, 1.0, 1.5, 2.5]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Power Law Transformations with Different γ Values', fontsize=16)

# og image
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

for i, gamma in enumerate(gamma_values):
    row = (i + 1) // 3
    col = (i + 1) % 3
    img_normalized = img / 255.0
    power_law_img = np.array(255 * (img_normalized ** gamma), dtype=np.uint8)
    axes[row, col].imshow(power_law_img, cmap='gray')
    axes[row, col].set_title(f'γ = {gamma}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

x = np.linspace(0, 255, 256)
plt.figure(figsize=(10, 6))
plt.title('Power Law Transformation Functions', fontsize=14)
plt.xlabel('Input Pixel Value', fontsize=12)
plt.ylabel('Output Pixel Value', fontsize=12)

for gamma in gamma_values:
    y = 255 * (x / 255.0) ** gamma
    plt.plot(x, y, label=f'γ = {gamma}')

plt.plot(x, x, '--', label='Identity (γ = 1)', color='black')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

stretched_img = histogram_stretch(img)


thresholded_img = threshold(img)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Piecewise Linear Intensity Transformations', fontsize=16)

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(stretched_img, cmap='gray')
axes[0, 1].set_title('Histogram Stretched')
axes[0, 1].axis('off')

axes[1, 0].imshow(thresholded_img, cmap='gray')
axes[1, 0].set_title('Thresholded')
axes[1, 0].axis('off')

x = np.linspace(0, 255, 256)
axes[1, 1].set_title('Transformation Functions')
axes[1, 1].set_xlabel('Input Pixel Value')
axes[1, 1].set_ylabel('Output Pixel Value')
low, high = np.percentile(img, [2, 98])
y_stretch = np.clip((x - low) * (255 / (high - low)), 0, 255)
axes[1, 1].plot(x, y_stretch, label='Histogram Stretch')

y_threshold = np.piecewise(x, 
                           [x < 100, (x >= 100) & (x < 200), x >= 200],
                           [0, lambda x: x, 255])
axes[1, 1].plot(x, y_threshold, label='Thresholding')
axes[1, 1].plot(x, x, '--', label='Identity', color='black')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Image Histograms', fontsize=16)

axes[0].hist(img.ravel(), bins=256, range=(0, 256))
axes[0].set_title('Original Image')
axes[0].set_xlabel('Pixel Value')
axes[0].set_ylabel('Frequency')

axes[1].hist(stretched_img.ravel(), bins=256, range=(0, 256))
axes[1].set_title('Histogram Stretched')
axes[1].set_xlabel('Pixel Value')

axes[2].hist(thresholded_img.ravel(), bins=256, range=(0, 256))
axes[2].set_title('Thresholded')
axes[2].set_xlabel('Pixel Value')

plt.tight_layout()
plt.show()




equalized_img, cdf_normalized = histogram_equalization(img)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Histogram Equalization', fontsize=16)

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(equalized_img, cmap='gray')
axes[0, 1].set_title('Histogram Equalized Image')
axes[0, 1].axis('off')

axes[1, 0].hist(img.ravel(), bins=256, range=(0, 256))
axes[1, 0].set_title('Original Histogram')
axes[1, 0].set_xlabel('Pixel Value')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(equalized_img.ravel(), bins=256, range=(0, 256))
axes[1, 1].set_title('Equalized Histogram')
axes[1, 1].set_xlabel('Pixel Value')

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.title('Histogram Equalization Transformation Function', fontsize=14)
plt.xlabel('Input Pixel Value', fontsize=12)
plt.ylabel('Output Pixel Value', fontsize=12)
plt.plot(np.arange(256), cdf_normalized, label='CDF Normalized', color='blue')
plt.plot(np.arange(256), np.arange(256), '--', label='Identity', color='black')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


breakpoints = [100, 200]
segments = [
    (0, 0), 
    (1, 0),  
    (0, 255)  
]


x = np.arange(256)
y_piecewise = piecewise_linear_transformation_function(x, breakpoints, segments)

# Plot the transformation functions
plt.figure(figsize=(10, 6))
plt.title('Comparison of Transformation Functions', fontsize=14)
plt.xlabel('Input Pixel Value', fontsize=12)
plt.ylabel('Output Pixel Value', fontsize=12)

plt.plot(x, cdf_normalized, label='Histogram Equalization CDF', color='blue')
plt.plot(x, y_piecewise, label='Piecewise Linear Transformation', color='red')
plt.plot(x, x, '--', label='Identity', color='black')

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# example value
gamma = 0.6 
lookup_table_power_law = power_law_lookup_table(gamma)
power_law_img = lookup_table_power_law[img]

plt.figure(figsize=(8, 6))
plt.imshow(power_law_img, cmap='gray')
plt.title(f'Power Law Transformation using Lookup Table (γ = {gamma})')
plt.axis('off')
plt.show()


lookup_table_piecewise = piecewise_linear_lookup_table(breakpoints, segments)
thresholded_img_lookup = lookup_table_piecewise[img]

plt.figure(figsize=(8, 6))
plt.imshow(thresholded_img_lookup, cmap='gray')
plt.title('Thresholded Image using Lookup Table')
plt.axis('off')
plt.show()