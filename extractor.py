from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

"""
We can use a pre-trained network to extract meaningful features from the extracted
frames. The [`Keras Applications`](https://keras.io/api/applications/) module provides
a number of state-of-the-art models pre-trained on the [ImageNet-1k dataset](http://image-net.org/).
We will be using the [InceptionV3 model](https://arxiv.org/abs/1512.00567) for this purpose.
"""

class Extractor():
    def __init__(self, image_shape, weights = None):
        """ load pre-trained from imagenet InceptionV3, or load saved weighted if we re-train network
        """
        self.weights = weights # check model

        base_model = InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=image_shape,
        )

        inputs = Input(image_shape)
        preprocessed = preprocess_input(inputs)
        outputs = base_model(preprocessed)
                
        self.model = Model(inputs, outputs, name="feature_extractor")


    def extract(self, img):
        x = preprocess_input(img)
        # get the image feature using predict
        features = self.model.predict(x)
        if self.weights is None:
            # load imagenet network
            features = features[0]
        else:
            # for loaded network
            features = features[0]        

        return features

        