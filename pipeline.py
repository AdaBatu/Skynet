from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, image_size=(64, 64), grayscale=True):
        self.image_size = image_size
        self.grayscale = grayscale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for img in X:
            try:
                # Convert NumPy array to PIL Image if needed
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)

                if self.grayscale:
                    img = img.convert('L')  # Grayscale

                img = img.resize(self.image_size)
                arr = np.asarray(img, dtype=np.float32).flatten() / 255.0
                features.append(arr)

            except Exception as e:
                print(f"Error processing image: {e}")
                features.append(np.zeros(self.image_size[0] * self.image_size[1]))

        return np.array(features)
