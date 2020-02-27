# importing libraries
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('VID_20200218_174114.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video  file")


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    if ret:
        ad_frame = adjust_gamma(frame, gamma=0.5)
        # Display the resulting frame
        cv2.imshow('Frame', ad_frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
