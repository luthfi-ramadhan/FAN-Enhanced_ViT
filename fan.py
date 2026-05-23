import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import gelu  # The paper uses GELU
from tensorflow.keras.initializers import GlorotUniform


class FANLayer(Layer):
    def __init__(self, dp, dp_bar, activation=gelu, **kwargs):
        super(FANLayer, self).__init__(**kwargs)
        self.dp = dp
        self.dp_bar = dp_bar
        self.activation = activation

    def build(self, input_shape):
        d_in = input_shape[-1]

        self.Wp = self.add_weight(
            shape=(d_in, self.dp),
            initializer=GlorotUniform(),
            trainable=True,
            name='Wp'
        )

        self.Wp_bar = self.add_weight(
            shape=(d_in, self.dp_bar),
            initializer=GlorotUniform(),
            trainable=True,
            name='Wp_bar'
        )
        self.Bp_bar = self.add_weight(
            shape=(self.dp_bar,),
            initializer='zeros',
            trainable=True,
            name='Bp_bar'
        )

    def call(self, inputs):
        proj = tf.matmul(inputs, self.Wp)
        cos_proj = tf.math.cos(proj)
        sin_proj = tf.math.sin(proj)

        mlp_proj = tf.matmul(inputs, self.Wp_bar) + self.Bp_bar
        mlp_out = self.activation(mlp_proj)

        return tf.concat([cos_proj, sin_proj, mlp_out], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2 * self.dp + self.dp_bar)
