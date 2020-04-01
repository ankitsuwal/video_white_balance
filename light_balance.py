# importing libraries
import cv2
import numpy as np
from constant import LIGHT_BAL, VIDEO_STABLE, LIGHT_STABLE
from video_stabilization import VideoStabilization
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def balanced_frame(cap):
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # TODO: uncomment below line of code if you want to rotate frame by 180 degree 
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        if ret:
            if LIGHT_BAL:
                ad_frame = adjust_gamma(frame, gamma=0.5)
            if VIDEO_STABLE:
                ad_frame = VideoStabilization.stablize_frame()
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

if __name__ == '__main__':
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video  file")

    result = balanced_frame(cap)
