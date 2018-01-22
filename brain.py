
import tensorflow as tf
from keras.models import Sequential

class Brain(object):
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.model = Sequential()

class NeuralNetwork(Brain):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._set_private_members()
        self.input_ph = tf.placeholder(shape=[None] + list(self.env.get_state_shape()), dtype=tf.float32)

    def _set_private_members(self):
        self.env_state_shape = self.env.get_state_shape()
        self.env_action_bound = self.env.get_action_bound()
        self.env_action_number = self.env.get_num_action()
