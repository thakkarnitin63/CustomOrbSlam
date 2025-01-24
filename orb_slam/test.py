import os 
import cv2
from orb_slam.image_loader import ImageLoader

def main():
    sequence_path = os.path.expanduser('~/NitinWs/CustomOrbSlam/data/dataset/sequences/00/')

    image_loader = ImageLoader(sequence_path)

    frame_id=0
    
    try:

        image = image_loader.load_image(frame_id)
        timestamp = image_loader.get_timestamp(frame_id)

        window_title =f"Frame {frame_id} at {timestamp:.6f} seconds"
        cv2.imshow(window_title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__== '__main__':
    main()