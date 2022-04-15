import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from extractor import Extractor
import cv2
from videolibs import crop_center_square


"""
trick: Convert frame from BGR to RGB
image = image[:, :, [2,1,0]]
"""
def load_video(path, max_frames, resize):
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


"""
Finally, we can put all the pieces together to create our data processing utility.
Note: Here the Ellipsis object selected all dimensions. (three dots ...)
Accessing and slicing multidimensional Arrays/NumPy indexing.
.numpy(): convert a Tensor to Numpy array.
"""

def prepare_all_videos(df, root_dir, label_processor, img_size, max_seq_length, nb_features):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values    
    labels = label_processor(labels[..., None]).numpy()
    
    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = np.zeros(shape=(num_samples, max_seq_length), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, max_seq_length, nb_features), dtype="float32"
    )

    # init extractor model
    model = Extractor(image_shape=(img_size, img_size, 3))

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(
            path=os.path.join(root_dir, path), 
            max_frames=max_seq_length, 
            resize=(img_size, img_size)
        )
        
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = np.zeros(shape=(1, max_seq_length, ), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, max_seq_length, nb_features), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(max_seq_length, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = model.extract(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked
        
        # numpy.sequeeze() function is used when we want to remove single-dimensional entries from the shape of an array.

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    # # Save the sequence.
    # save_path = r'data\\tmp\\'
    # np.save(open(save_path + 'frame_features.npy', 'wb'), frame_features)
    # np.save(open(save_path + 'frame_mask.npy','wb'), frame_masks)
    # np.save(open(save_path + 'labels.npy','wb'), labels)

    return (frame_features, frame_masks), labels

def load_all_features(features_file, mask_file, lables_file):

    frame_features = np.load(open(features_file, mode='rb'))   
    frame_masks = np.load(open(mask_file, mode='rb'))
    labels = np.load(open(lables_file, mode='rb'))
    
    return (frame_features, frame_masks), labels

