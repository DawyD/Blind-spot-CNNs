from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from convhole import ConvHole2D


def get_blindspot_model(input_shape, out_channels, depth=11, kernel_initializer=None, bias_initializer=None):
    inputs = Input(shape=input_shape)
    basic_convs = [inputs]
    basic_rf = 0
    for c in range(depth - 1):
        res = Conv2D(128, 3, padding="same",
                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(basic_convs[-1])
        res = LeakyReLU(alpha=0.1)(res)
        basic_convs.append(res)
        basic_rf = (basic_rf + 2) if c != 0 else (basic_rf + 3)

    hole_rf = 0
    hole_convs = []
    for c in range(depth):
        res = ConvHole2D(18, 3, dilation_rate=c+1, padding="same",
                         kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(basic_convs[c])
        res = LeakyReLU(alpha=0.1)(res)
        hole_convs.append(res)
        hole_rf = (hole_rf + 4) if c != 0 else (hole_rf + 3)

    concat_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = Concatenate(axis=concat_axis)(hole_convs)

    # Output stages.

    x = Conv2D(198, 1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(99, 1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(99, 1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(out_channels, 1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(x)

    return Model(inputs=inputs, outputs=x)
