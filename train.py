from tensorflow import keras
import pandas as pd
import numpy as np
import os
import pickle
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import LSTM_Model
from prepare_data import load_all_features

EPOCHS = 10
DATA_PATH = r"data\\"
model = 'lstm'

NUM_FEATURES = 2048
MAX_SEQ_LENGTH = 20
IMG_SIZE = 229

"""
The labels of the videos are strings. Neural networks do not understand string values,
so they must be converted to some numerical form before they are fed to the model. Here
we will use the [`StringLookup`](https://keras.io/api/layers/preprocessing_layers/categorical/string_lookup)
layer encode the class labels as integers.
"""

def get_class_one_hot(vocabs):

    label_processor = keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.unique(vocabs)
    )       
    return label_processor

def main():

    train_df = pd.read_csv(DATA_PATH + "train.csv")
    test_df = pd.read_csv(DATA_PATH + "test.csv")

    print(f"Total videos for training: {len(train_df)}")
    print(f"Total videos for testing: {len(test_df)}")

    label_processor = get_class_one_hot(train_df["tag"].values)

    feature_path = r"data\tmp"
    train_data, train_labels = load_all_features(
        features_file= os.path.join(feature_path, 'train-frame_features.npy'),
        mask_file= os.path.join(feature_path, 'train-frame_mask.npy'),
        lables_file= os.path.join(feature_path, 'train-labels.npy'),
    )

    test_data, test_labels = load_all_features(
        features_file= os.path.join(feature_path, "test-frame_features.npy"),
        mask_file= os.path.join(feature_path, "test-frame_mask.npy"),
        lables_file= os.path.join(feature_path, "test-labels.npy"),
    )
   

    print(f"Frame features in train set: {train_data[0].shape}")
    print(f"Frame masks in train set: {train_data[1].shape}")
    
    
    # Helper: Save the model.    
    checkpoints_dir = r"data\\checkpoints\\"

    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoints_dir,
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )
    
    class_vocab = label_processor.get_vocabulary()
    # Get the model.
    lstm  = LSTM_Model(
        nb_classes=len(class_vocab), 
        nb_features= NUM_FEATURES, 
        max_seq_length= MAX_SEQ_LENGTH
    )
        
    history = lstm.model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    lstm.model.load_weights(checkpoints_dir)
    _, accuracy = lstm.model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f"Test accuracy: {np.round(accuracy * 100, 2)}%")

    # serialize the model to disk
    print("[INFO] serializing network...")
    model_name = os.path.join(checkpoints_dir, "lstm.model")   
    lstm.model.save(model_name, save_format="h5")
    # serialize the label binarizer to disk
    model_label = os.path.join(checkpoints_dir, "tag.pickle")
    f = open(model_label, "wb")
    f.write(pickle.dumps(class_vocab))
    f.close()

    # Helper: TensorBoard
    # tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    # early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    # timestamp = time.time()
    # csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
    #     str(timestamp) + '.log'))
    


if __name__ == '__main__':
    main()
