import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import GRU, Dropout, Dense, Masking, LSTM
from keras import Input, Model
from tensorflow.keras.optimizers import Adam, RMSprop


class LSTM_Model():
    def __init__(self, nb_classes, nb_features, max_seq_length):
        """
        model = lstm
        nb_classes: the number of classes to predict
        nb_features: the number of image features. e.g., 2048 for InceptionV3
        max_seq_length: the maximum of frames per video
        """
        # set defaults
        self.nb_classes = nb_classes 
        self.nb_features = nb_features
        self.max_seq_length = max_seq_length
        self.model = self.lstm()

        print(self.model.summary())

    def lstm(self):    
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model."""
        
        frame_features_input = keras.Input((self.max_seq_length, self.nb_features))
        mask_input = keras.Input((self.max_seq_length,), dtype="bool")
        
        # Refer to the following tutorial to understand the significance of using `mask`:
        # https://keras.io/api/layers/recurrent_layers/gru/

        x = GRU(2048, return_sequences=True)(frame_features_input, mask=mask_input)     
        x = Dropout(0.4)(x)
        x = Dense(1024, activation='relu')(x)
        x = GRU(1024)(x)
        x = Dropout(0.4)(x)
        x = Dense(512, activation="relu")(x)

        output = keras.layers.Dense(self.nb_classes, activation="softmax")(x)

        model = keras.Model([frame_features_input, mask_input], output)

        # set metrics. 
        metrics = ['accuracy']  
        # Now compile the network.
        optimizer = Adam(learning_rate=1e-5, decay=1e-6)
        model.compile(
            loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=metrics
        )

        return model