# custom_initializers.py
import tensorflow as tf
import numpy as np

class OrthogonalInitializer(tf.keras.initializers.Initializer):
    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype=None):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.RandomState(seed=self.seed).normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return self.gain * q

    def get_config(self):
        return {"gain": self.gain, "seed": self.seed}