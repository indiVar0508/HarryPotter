import numpy as np
import cv2

#Capture The Camera
cam = cv2.VideoCapture(0)
#Background Image
background = cv2.resize(cv2.imread('SRC\\Filter.jpg'), (1280, 720))
#cv2.imshow('Background', background)
pixel = np.array([0, 200, 120]) #used find the cloak initial color
hsv = None

# lower = np.array([63, 201, 68])
# upper = np.array([123, 251, 198])

#the range of value
lower = np.array([23, 171, 58])
upper = np.array([133, 291, 208])


# lower = np.array([34, -15, 80])
# upper = np.array([84, 30, 192])

# lower = np.array([83, 51, 97])
# upper = np.array([103, 71, 177]) 

#function to detect color
def pick_color(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
	    pixel = hsv[y,x]
	    #you might want to adjust the ranges(+-10, etc):
	    upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
	    lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
	    print(pixel, lower, upper)
	    image_mask = cv2.inRange(hsv,lower,upper)
	    cv2.imshow("mask",image_mask)

while True:

	_, frame = cam.read()
	frame = cv2.resize(frame, (1280, 720))
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert to hsv
	mask = cv2.inRange(hsv, lower, upper) # mask the cloak
	kernel = np.ones((5,5),np.uint8) #kernel for smoothning the mask
	mask1 = cv2.dilate(mask,kernel,iterations = 1) #super sensitive
	res = cv2.bitwise_and(background, background, mask = mask1) #merge background with mask
	output = cv2.add(frame, res) #add updated mask with original frame
	# cv2.imshow('Frame', frame)
	# cv2.imshow('HSV', hsv)
	# cv2.imshow('Mask', mask)
	# cv2.imshow('Mask', mask1)
	# cv2.imshow('Res', res)
	cv2.imshow('output', output)
	#Uncomment if you wanna customize your own color or Cloak to detect proper pixel range
	# cv2.namedWindow('hsv')
	# cv2.setMouseCallback('hsv', pick_color)
	

	if cv2.waitKey(5) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()
cam.release()
