import numpy
import os

PARAMS_PICKLED = os.path.join(os.path.dirname(__file__), '..', 'data', 'nn.pickle')


def predict(params, batch_input, dropout=False, np=numpy):
    layer = batch_input
    for W, h in params[:-1]:
        layer = np.dot(layer, W) + h
        layer = np.maximum(layer, 0.03*layer)  # leaky relu
        if dropout:
            layer = dropout * 2 * np.random.randint(0, 2, size=layer.shape)
    return layer
