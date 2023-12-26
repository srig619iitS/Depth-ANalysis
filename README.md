Basic Depth Analysis using Inversion of intensity Technique: 
Assumes: Higher pixel intensity values on the gray scale images correspond to objects closer to camera and vice versa.
Inverting the intensity value technique is a commonly known technique used for image visualization of depth analysis using a OpenCV library,basically it subtracts gray scale image pixels from 255  to create a depth map. Other popular techniques like stereo vision and machine learning based approaches.
Once after creation of depth map, resize the depth map to match the dimensions of the original color image . Later combine the original image and depth map, the resulting image is a mix of color image and the depth map, emphasizing depth information.The final image shows the color information from the original image combined with the depth information from the generated depth map.
basic Depth Map generation steps:
# load Image
	import cv2
	rover_image = cv2.imread("/Image.png")  
# Convert the image to grayscale
	gray_image = cv2.cvtColor(rover_image, cv2.COLOR_BGR2GRAY)
# Apply a simple depth map technique (inverse of intensity)
	depth_map = 255 - gray_image
# Resize the depth map to match the original image
	depth_map_resized = cv2.resize(depth_map, (rover_image.shape[1], 	rover_image.shape[0]))
# Combine the color image and depth map
	combined_image = cv2.addWeighted(rover_image, 0.7, 	cv2.cvtColor(depth_map_resized, cv2.COLOR_GRAY2BGR), 0.3, 0)
# Display the result
	plt.imshow(combined_image)
	plt.title('Simple Depth Analysis of Rover Image')
	plt.show()


--------------------------------------------------------------------------------------------------------
Capture Image details
	from PIL import Image
	import numpy as np
	import matplotlib.pyplot as plt
	import matplotlib.image as mpimg

# Open the image
	im = Image.open("/Image.png")

# Get the pixel values
	pixels = im.load()

width, height = im.size
	print(width, height)

#unreveling
	arr = np.array(pixels)
	print(arr)

#Convert 2D array to 1D array
	arr_1d = arr.ravel()
	print(arr_1d)

	imgplot = plt.imshow(im)
	plt.show()


--------------------------------------------------------------------------------------------------------------------------

Image thresholding and Edge detection Techniques on an image

	from transformers import pipeline
	from PIL import Image
	import cv2
	import numpy as np
	from matplotlib import pyplot as plt
#open the image
	im = Image.open("/Object.jpg")
	image = np.array(im)
	image
# 1. Simple Thresholding
_, simple_threshold = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
print("Simple Threshold Value:", 128)

# 2. Otsu's Binarization
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, otsu_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu's Threshold Value:", _)

# 3. Adaptive Thresholding
adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
print("Adaptive Thresholding - Gaussian C: 11x11, Constant 2")
# Original Image and Histogram
	plt.subplot(3, 2, 1), plt.imshow(image, cmap='gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(3, 2, 2), plt.hist(image.ravel(), 256, [0, 256], color='r', alpha=0.75)
	plt.title('Original Image Histogram')

# Simple Thresholding and Histogram
	plt.subplot(3, 2, 3), plt.imshow(simple_threshold, cmap='gray')
	plt.title('Simple Thresholding'), plt.xticks([]), plt.yticks([])
	plt.subplot(3, 2, 4), plt.hist(simple_threshold.ravel(), 256, [0, 256], color='r', alpha=0.75)
	plt.title('Simple Thresholding Histogram')

# Otsu's Binarization and Histogram
	plt.subplot(3, 2, 5), plt.imshow(otsu_threshold, cmap='gray')
	plt.title("Otsu's Binarization"), plt.xticks([]), plt.yticks([])
	plt.subplot(3, 2, 6), plt.hist(otsu_threshold.ravel(), 256, [0, 256], color='r', alpha=0.75)
	plt.title("Otsu's Binarization Histogram")

	plt.tight_layout()
	plt.show()

--------------------------------------------------------------------------------------------------------------------
Prewitt operator: 
#Edge detection
	from transformers import pipeline
	from PIL import Image
	import cv2
	import numpy as np
	from matplotlib import pyplot as plt

#open the image
	im = cv2.imread("/Object.jpg")

# Convert the image to grayscale if it's in RGB format
	gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Prewitt operator kernels - prewitt_kx and prewitt_ky are kernels used for Prewitt edge detection.
#These kernels are convolution filters that are applied to the image to compute the gradient in the horizontal (x)
# and vertical (y) directions
	prewitt_kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
	prewitt_ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

# Apply Prewitt operators - obtain the gradient images in the x and y directions,
# These gradient images represent the rate of change of intensity in the corresponding directions.
	prewitt_x = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_kx)
	prewitt_y = cv2.filter2D(gray_image, cv2.CV_64F, prewitt_ky)

# Compute the magnitude of the gradient - The magnitude of the gradient is calculated by taking
# the absolute values of the gradients in the x and y directions and summing them element-wise.
	prewitt_magnitude = np.abs(prewitt_x) + np.abs(prewitt_y)
# Display the results
	plt.figure(figsize=(12, 6))
	plt.subplot(1, 4, 1), plt.imshow(gray_image, cmap='gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# Add this line to display the grayscale image
	plt.subplot(1, 4, 2), plt.imshow(prewitt_x, cmap='gray')
	plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
	plt.subplot(1, 4, 3), plt.imshow(prewitt_y, cmap='gray')
	plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])
	plt.subplot(1, 4, 4), plt.imshow(prewitt_magnitude, cmap='gray')
	plt.title('Prewitt Magnitude'), plt.xticks([]), plt.yticks([])
	plt.show()
