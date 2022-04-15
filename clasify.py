import os
import cv2
import numpy as np
from extractor import Extractor
from keras.models import load_model
import pickle
from videolibs import crop_center_square


IMG_SIZE = 229
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

video_file = r"demo.avi"
cap = cv2.VideoCapture(video_file)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_writer = cv2.VideoWriter("result.avi", fourcc, 15, (int(width), int(height)))

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")

feature_extractor = Extractor(image_shape=(IMG_SIZE, IMG_SIZE, 3))
saved_model = r"data\\checkpoints\\lstm.model"
saved_labels = r"data\\checkpoints\\tag.pickle"
lstm_model = load_model(saved_model)

with open(saved_labels, "rb") as f:
    rawdata = f.read()    
labels = pickle.loads(rawdata)
print(labels)

frame_count = 0
frames = []
original_frames = []
while True:
    ret, frame = cap.read()
    # Bail out when the video file ends
    if not ret:
        break
    original_frames.append(frame)
    frame = crop_center_square(frame)
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame[:, :, [2, 1, 0]]
   
    # Save each frame of the video to a list
    frame_count += 1
    frames.append(frame)
    
    if frame_count < MAX_SEQ_LENGTH:
        continue # capture frames untill you get the required number for sequence
    else:
        frame_count = 0

    sequences = np.array(frames)    
    sequences = sequences[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # For each frame extract feature and prepare it for classification

    for i, batch in enumerate(sequences):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.extract(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    # Clasify sequence    
    probabilities = lstm_model.predict([frame_features, frame_mask])[0]
    labels_idx = probabilities.argmax(axis=-1)
    result = ""
    # if probabilities[labels_idx] > 0.5:
    result = f"{labels[labels_idx]}: {probabilities[labels_idx] * 100:5.2f}%"

    # for i in np.argsort(probabilities)[::-1]:        
    #     # print(f"  {labels[i]}: {probabilities[i] * 100:5.2f}%")
    #     result.append(f"  {labels[i]}: {probabilities[i] * 100:5.2f}%")

    # Add prediction to frames and write them to new video
    for image in original_frames:
        img = np.ascontiguousarray(image, dtype=np.uint8)       
        cv2.putText(img, result, (40, 40 * i + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.imshow('window', img)
        # when 'q' key is pressed the video capture stops
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frames = []
    original_frames = []
    
cv2.destroyAllWindows()