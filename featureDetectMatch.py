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


def match_features(descriptors1, descriptors2):
    # Initialize the Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #Match descriptors
    matches = bf.match(descriptors1,descriptors2)

    #sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    return matches

def visualize_keypoints(image,keypoints):
    # draw keypoints on image
    image_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, color=(0,255,0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )


    # Create a resizable window
    cv2.namedWindow('ORB Keypoints', cv2.WINDOW_NORMAL)

    # Set the window size (e.g., 1980x1080)
    cv2.resizeWindow('ORB Keypoints', 1980, 1080)

    # Display the image
    cv2.imshow('ORB Keypoints', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_path1= os.path.join(script_directory, 'SampleImages', 'buddha_001.png')
    image_path2= os.path.join(script_directory, 'SampleImages', 'buddha_002.png')    
    # keypoints, descriptors, image =extract_orb_features(image_path)
    # visualize_keypoints(image,keypoints)

    #Extract Features from both images
    kp1, des1, img1 = extract_orb_features(image_path1)
    kp2, des2, img2 = extract_orb_features(image_path2)

    #match features
    matches = match_features(des1, des2)

    #Draw matches
    matched_image = cv2.drawMatches(img1,kp1,img2,kp2, matches[:50], None, flags=2)

    # Create a resizable window
    cv2.namedWindow('Feature Matches', cv2.WINDOW_NORMAL)

    # Set the window size (e.g., 1980x1080)
    cv2.resizeWindow('Feature Matches', 1980, 1080)

    #display the matched image
    cv2.imshow('Feature Matches', matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



