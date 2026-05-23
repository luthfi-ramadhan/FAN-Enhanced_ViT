import numpy as np
import keras
from keras import layers
import tensorflow as tf
from fan import FANLayer
n_classes = 1


class PositionEncoding(layers.Layer):
    def __init__(self, num_seq, projection_dim):
        super(PositionEncoding, self).__init__()
        self.num_seq = num_seq
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_seq, output_dim=projection_dim
        )

    def call(self, seq):
        positions = tf.range(start=0, limit=self.num_seq, delta=1)
        encoded = self.projection(seq) + self.position_embedding(positions)
        return encoded
    

    
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    print(res.shape)
    dp_ = int(0.25 * ff_dim)
    dp_bar_ = int(ff_dim - (2 * dp_))
    
    # Feed Forward Part
    x = FANLayer(dp=dp_, dp_bar=dp_bar_)(res)
    print(x.shape)
    x = layers.Dropout(dropout)(x)
    x = FANLayer(dp=dp_, dp_bar=dp_bar_)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)
    seq, f_ = input_shape
    x = inputs

    x = PositionEncoding(seq, 512)(x)

    
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        dp_ = int(0.25 * dim)
        dp_bar_ = int(dim - (2 * dp_))
        x = FANLayer(dp=dp_, dp_bar=dp_bar_)(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

