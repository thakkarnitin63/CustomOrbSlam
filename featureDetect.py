import cv2
import numpy as np
import os

def extract_orb_features(image_path, nfeatures=500):
    #Load Image
    image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' could not be load. ")
    
    #initialize ORB detector
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Detect Keypoint and compute Descriptors
    keypoints, descriptors = orb.detectAndCompute(image,None)

    return keypoints, descriptors, image

def visualize_keypoints(image,keypoints):
    # draw keypoints on image
    image_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, color=(0,255,0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )


    # Create a resizable window
    cv2.namedWindow('ORB Keypoints', cv2.WINDOW_NORMAL)

    # Set the window size (e.g., 800x600)
    cv2.resizeWindow('ORB Keypoints', 1980, 1080)

    # Display the image
    cv2.imshow('ORB Keypoints', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_directory, 'SampleImages', 'buddha_001.png')    
    keypoints, descriptors, image =extract_orb_features(image_path)
    visualize_keypoints(image,keypoints)

    