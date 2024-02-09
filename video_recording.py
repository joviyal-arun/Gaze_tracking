import cv2


# Create an object to read
# from camera
video = cv2.VideoCapture(0)

# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
	print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('video_input/20_6_2022/head_straight/head_straight_eye_left_30.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
	
while(True):
	ret, frame = video.read()

	if ret == True:

		# Write the frame into the
		# file 'filename.avi'
		result.write(frame)

		# Display the frame
		# saved in the file
		cv2.imshow('Frame', frame)

		# Press S on keyboard
		# to stop the process
		if cv2.waitKey(1) & 0xFF == ord('s'):
			break

	# Break the loop
	else:
		break

# When everything done, release
# the video capture and video
# write objects
video.release()
result.release()
	
# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")


'''
# Program To Read video
# and Extract Frames

import cv2

# Function to extract frames
def FrameCapture(path):

	# Path to video file
	vidObj = cv2.VideoCapture(path)

	# Used as counter variable
	count = 0

	# checks whether frames were extracted
	success = 1

	while success:

		# vidObj object calls read
		# function extract frames
		success, image = vidObj.read()

		# Saves the frames with frame-count
		cv2.imwrite("video_output/frame_by_frame_output/frame%d.jpg" % count, image)

		count += 1


# Driver Code
if __name__ == '__main__':

	# Calling the function
	FrameCapture("video_input/20_6_2022/head_straight/head_straight_eye_straight.avi")
'''