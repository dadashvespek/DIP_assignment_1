from libtiff import TIFF
from matplotlib import pyplot as plt
import numpy as np

tif = TIFF.open('imageset4/Ghost32.tif', mode='r')
image = tif.read_image()
tif.close()
height, width = image.shape
pixel_size = 25.4 / 300 # mm/pixel (300 pixels per inch, 1 inch = 25.4 mm)

def index_to_local(i, j):
    x = j * pixel_size
    y = (height - 1 - i) * pixel_size
    return x, y

def local_to_world(x, y):
    # we'll assume a simple translation
    world_x = x + 10  
    world_y = y + 5   
    return world_x, world_y

def rotation_matrix(angle_degrees):
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def scaling_matrix(sx, sy):
    return np.array([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, 1]])

def translation_matrix(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])

def compose_transformations(*transforms):
    result = np.eye(3)
    for t in transforms:
        result = np.dot(result, t)
    return result


def apply_transformation(image, transform_matrix, interpolation='nearest'):
    height, width = image.shape
    new_image = np.zeros_like(image)
    
    inv_transform = np.linalg.inv(transform_matrix)
    
    for y in range(height):
        for x in range(width):
            orig_x, orig_y, _ = np.dot(inv_transform, [x, y, 1])
            
            if 0 <= orig_x < width-1 and 0 <= orig_y < height-1:
                if interpolation == 'nearest':
                    new_image[y, x] = nearest_neighbor_interpolation(image, orig_x, orig_y)
                elif interpolation == 'bilinear':
                    new_image[y, x] = bilinear_interpolation(image, orig_x, orig_y)
            # else:
            #     new_image[y, x] = 0  # Set to black for out-of-bounds pixels
    
    return new_image

def nearest_neighbor_interpolation(image, x, y):
    return image[min(int(round(y)), image.shape[0]-1), min(int(round(x)), image.shape[1]-1)]

def bilinear_interpolation(image, x, y):
    x1, y1 = int(x), int(y)
    x2, y2 = min(x1 + 1, image.shape[1] - 1), min(y1 + 1, image.shape[0] - 1)
    
    fq11 = image[y1, x1]
    fq21 = image[y1, x2]
    fq12 = image[y2, x1]
    fq22 = image[y2, x2]
    
    return (fq11 * (x2 - x) * (y2 - y) +
            fq21 * (x - x1) * (y2 - y) +
            fq12 * (x2 - x) * (y - y1) +
            fq22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1))
def apply_and_visualize_transformation(image, transform, title):
    nearest_result = apply_transformation(image, transform, 'nearest')
    bilinear_result = apply_transformation(image, transform, 'bilinear')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')

    ax2.imshow(nearest_result, cmap='gray')
    ax2.set_title(f'{title}\nNearest Neighbor')

    ax3.imshow(bilinear_result, cmap='gray')
    ax3.set_title(f'{title}\nBilinear')

    plt.tight_layout()
    plt.show()

rotation = rotation_matrix(30)  
scaling = scaling_matrix(1.5, 1.5)  
translation = translation_matrix(10, 20)  

composite_transform = compose_transformations(translation, rotation, scaling)

def apply_transform(matrix, x, y):
    point = np.array([x, y, 1])
    transformed = np.dot(matrix, point)
    return transformed[0], transformed[1]

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image, cmap='gray')

indices = [(0, 0), (0, width-1), (height-1, 0), (height-1, width-1)]
for i, j in indices:
    x, y = index_to_local(i, j)
    wx, wy = local_to_world(x, y)
    ax.plot(j, i, 'ro') 
    ax.annotate(f'({i},{j})\n({x:.1f},{y:.1f})\n({wx:.1f},{wy:.1f})', (j, i), 
                xytext=(5, 5), textcoords='offset points', color='red', fontsize=8)

ax.set_title('Ghost32 Image with Coordinate Systems')
plt.show()

print("Example coordinate transformations:")
for i, j in indices:
    x, y = index_to_local(i, j)
    wx, wy = local_to_world(x, y)
    print(f"Image index: ({i}, {j})")
    print(f"Local coordinates: ({x:.2f}, {y:.2f})")
    print(f"World coordinates: ({wx:.2f}, {wy:.2f})")
    print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')

ax2.imshow(image, cmap='gray')
ax2.set_title('Transformed Image (Corner Points)')

corners = [(0, 0), (0, width-1), (height-1, 0), (height-1, width-1)]
for corner in corners:
    i, j = corner
    x, y = index_to_local(i, j)
    tx, ty = apply_transform(composite_transform, x, y)
    ax1.plot(j, i, 'ro')
    ax2.plot(tx/pixel_size, height - ty/pixel_size, 'ro')
    ax2.annotate(f'({tx:.1f}, {ty:.1f})', (tx/pixel_size, height - ty/pixel_size), 
                 xytext=(5, 5), textcoords='offset points', color='red', fontsize=8)

plt.show()

print("Example composite transformations:")
for i, j in corners:
    x, y = index_to_local(i, j)
    tx, ty = apply_transform(composite_transform, x, y)
    print(f"Original local coordinates: ({x:.2f}, {y:.2f})")
    print(f"Transformed coordinates: ({tx:.2f}, {ty:.2f})")
    print()




rotation = rotation_matrix(30)
scaling = scaling_matrix(0.7, 0.7)
translation = translation_matrix(0, 0) 

composite_transform = compose_transformations(translation, rotation, scaling)

nearest_neighbor_result = apply_transformation(image, composite_transform, 'nearest')
bilinear_result = apply_transformation(image, composite_transform, 'bilinear')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')

ax2.imshow(nearest_neighbor_result, cmap='gray')
ax2.set_title('Nearest Neighbor Interpolation')

ax3.imshow(bilinear_result, cmap='gray')
ax3.set_title('Bilinear Interpolation')

plt.tight_layout()
plt.show()



# 1. Rotation
rotation_transform = rotation_matrix(45)
apply_and_visualize_transformation(image, rotation_transform, 'Rotation (45°)')

# 2. Scaling
scaling_transform = scaling_matrix(1.5, 0.75)
apply_and_visualize_transformation(image, scaling_transform, 'Scaling (1.5x horizontal, 0.75x vertical)')

# 3. Translation
translation_transform = translation_matrix(5, 5)
apply_and_visualize_transformation(image, translation_transform, 'Translation (5 pixels right and down)')

# 4. Shear
shear_transform = np.array([[1, 0.5, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
apply_and_visualize_transformation(image, shear_transform, 'Shear (horizontal)')

composite_transform = compose_transformations(
    rotation_matrix(30),
    scaling_matrix(1.2, 1.2),
    translation_matrix(2, 2)
)
apply_and_visualize_transformation(image, composite_transform, 'Composite (Rotation, Scaling, Translation)')