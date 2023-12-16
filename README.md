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
