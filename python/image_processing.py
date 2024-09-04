import numpy as np 
import matplotlib.cbook as cbook 
import matplotlib.pyplot as plt 
from skimage import morphology # pip install scikit-image
from skimage.morphology import square
import cv2  # pip install opencv-python 
from skimage import measure
import pandas as pd
import os
import sys


with cbook.get_sample_data(os.path.join(os.path.dirname(__file__), '../year2000.jpg')) as image_file_2000:
    image2000 = plt.imread(image_file_2000)
with cbook.get_sample_data(os.path.join(os.path.dirname(__file__), '../year2016.jpg')) as image_file_2016:
    image2016 = plt.imread(image_file_2016)


# Display the image from 2000.
fig, ax = plt.subplots()
ax.imshow(image2000)
ax.axis('off')
plt.title('Year 2000')


# Compare to 2016
fig, (ax1,ax2) = plt.subplots(1,2,layout='constrained') 
ax1.imshow(image2000) 
ax1.axis('off') 
ax2.imshow(image2016) 
ax2.axis('off')
fig.suptitle('Year 2000 vs Year 2016')


# Convert to grayscale
if image2016.ndim == 3:  
    gray2016 = np.dot(image2016[..., :3], [0.2989, 0.5870, 0.1140]) 
    gray2016 = np.round(gray2016).astype(int)
fig, ax = plt.subplots()
ax.imshow(gray2016, cmap='gray', vmin=0, vmax=255)
ax.axis('off')
plt.title('2016 in grayscal')


# Display histogram
fig, ax = plt.subplots()
pixel_values = gray2016.flatten()
ax.hist(pixel_values, bins=256, rwidth=0.3)
plt.title('Grayscale histogram')


# Adjust contrast
lower_bound = np.percentile(pixel_values,1)
upper_bound = np.percentile(pixel_values,99)
gray2016_saturated = np.clip(pixel_values, lower_bound, upper_bound)
gray2016_rescaled = 255 * (gray2016_saturated - lower_bound) / (upper_bound - lower_bound)
gray2016_rescaled = gray2016_rescaled.astype(np.uint8)
fig, ax = plt.subplots()
ax.hist(gray2016_rescaled, bins=256, rwidth=0.3)
plt.title('Adjusted histogram')


# Compare and contrast
fig, (ax1,ax2) = plt.subplots(1,2,layout='constrained') 
ax1.imshow(gray2016, cmap='gray', vmin=0, vmax=255) 
ax1.axis('off') 
unflattened_imag = gray2016_rescaled.reshape(800,720)
ax2.imshow(unflattened_imag,cmap='gray', vmin=0, vmax=255) 
ax2.axis('off')
fig.suptitle('Before and after adjustment')


# Import the image from 2004
with cbook.get_sample_data(os.path.join(os.path.dirname(__file__), '../year2004.jpg')) as image_file_2004:
    image2004 = plt.imread(image_file_2004)

def im2gray(image2004):
    if image2004.ndim == 3:  
        gray2004 = np.dot(image2004[..., :3], [0.2989, 0.5870, 0.1140]) 
        gray2004 = np.round(gray2004).astype(int)
    return gray2004
gray2004 = im2gray(image2004)

def imadjust(gray2004):
    pixel_values = gray2004.flatten()
    lower_bound = np.percentile(pixel_values,1)
    upper_bound = np.percentile(pixel_values,99)
    gray_saturated = np.clip(pixel_values, lower_bound, upper_bound)
    gray_rescaled = 255 * (gray_saturated - lower_bound) / (upper_bound - lower_bound)
    gray_rescaled = gray_rescaled.astype(np.uint8)
    adj_image = gray_rescaled.reshape(800,720)
    threshold = 100
    adj_image = adj_image > threshold
    adj_image = adj_image.astype(int)

    selem = square(3)
    morph_image = morphology.opening(adj_image, selem)
    return morph_image 

adj2004 = imadjust(gray2004)


# Try color thresholding on 2004
def create_mask(RGB):
    # Convert RGB image to HSV color space
    I = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)

    # Define thresholds for HSV channels based on createMask.m file
    channel1Min = 0.000
    channel1Max = 0.213
    channel2Min = 0.000
    channel2Max = 1.000
    channel3Min = 0.000
    channel3Max = 0.791

    # Create mask based on thresholds based on createMask.m file
    mask = (I[:,:,0] >= channel1Min * 180) & (I[:,:,0] <= channel1Max * 180) & \
           (I[:,:,1] >= channel2Min * 255) & (I[:,:,1] <= channel2Max * 255) & \
           (I[:,:,2] >= channel3Min * 255) & (I[:,:,2] <= channel3Max * 255)
    BW = mask.astype(np.uint8)   
    return BW

bw2004 = create_mask(image2004)
fig, axes = plt.subplots(1, 2)
axes[0].imshow(image2004)
axes[0].axis('off')
axes[1].imshow(bw2004,cmap='gray', vmin=0, vmax=1)
axes[1].axis('off')
plt.suptitle("2004 with Color Thresholding")



# Calculate the area in pixels
def calculate_area_pixel(bwImage):
    properties = []
    props = measure.regionprops(measure.label(bwImage))
    areaPixels = sum(prop.area for prop in props)
    for prop in props:
        properties.append({
            'Label': prop.label,
            'Area': prop.area,
        })
    return pd.DataFrame(properties), areaPixels

df, areaPixels = calculate_area_pixel(bw2004)
print(df)
print(f"areaPixels = {areaPixels}")

# Convert pixels to square kilometers
def pixel2km(areaPixels):
    px2km = round((51/20)**2, 4)
    areaKmSq = int(areaPixels*px2km)
    return px2km, areaKmSq

px2km, areaKmSq = pixel2km(areaPixels)
print(f"px2km = {px2km}")
print(f"areaKmSq  = {areaKmSq}")


years = ['2000', '2004', '2008', '2012', '2016']



#   Plot area vs year as a bar plot.
areasKmSq = []
for year in years:
    with cbook.get_sample_data(os.path.join(os.path.dirname(__file__), f'../year{year}.jpg')) as image_file:
        image = plt.imread(image_file)
    gray = im2gray(image)
    adj = imadjust(gray)
    bw = create_mask(image)
    areaPixels = calculate_area_pixel(bw)[1]
    areaKmSq = pixel2km(areaPixels)[1] / (10**5)
    print(year,areaKmSq)
    areasKmSq.append(areaKmSq)

fig, axe = plt.subplots()
axe.bar(years,areasKmSq)
axe.set(ylabel=' 10^5 km^2', xlabel='Year', title='Deforested areas in km^2')


# Display all segmented images
fig, axes = plt.subplots(int(np.ceil(len(years)/3)), 3, layout='constrained')

i = 0
j = 0
for year in years:
    with cbook.get_sample_data(os.path.join(os.path.dirname(__file__), f'../year{year}.jpg')) as image_file:
        image = plt.imread(image_file)
    gray = im2gray(image)
    adj = imadjust(gray)
    bw = create_mask(image)
    areasKmSq.append(areaKmSq)
    axes[i][j].imshow(bw,cmap='gray', vmin=0, vmax=255)
    axes[i][j].axis('off')

    j += 1
    if j >= 3:
        i += 1
        j = 0

plt.suptitle("All segmented images")

plt.show()












# import scipy.io
# data = scipy.io.loadmat('/Users/macmini/Desktop/projects/getting-started-with-image-processing/python/bw2004.mat')
# matrix_name = data['bw2004']

# are_equal = np.array_equal(matrix_name, bw2004)
# print(are_equal)

# differences = np.subtract(matrix_name, bw2004)
# difference_locations = np.where(differences != 0)
# print(difference_locations)

# np.set_printoptions(threshold=sys.maxsize)
# file = open("file1.txt", "w+")
 
# content = str(differences)
# file.write(content)
# file.write(str(difference_locations))
# file.close()
