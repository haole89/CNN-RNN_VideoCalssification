from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

IMG_SIZE = 229

def process_image(image, target_shape):

    """Given an image, process it and return the array."""
    
    # Load the image
    h, w, _ = target_shape
    image = load_img(image, target_size=(h, w))

    # Convert it into numpy, normalize and return
    img_arr = img_to_array(image)
    x = (img_arr / 255.).astype(np.float32)

    return x

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

"""
trick: Convert frame from BGR to RGB
image = image[:, :, [2,1,0]]
"""
def load_video(path, max_frames = 0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)