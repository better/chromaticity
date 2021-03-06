from autograd import numpy, grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam
import h5py
import pickle
import random
from colors.nn import predict, PARAMS_PICKLED


def predict_f(*args, **kwargs):
    # Inject autograd's numpy definitions instead
    kwargs.update(dict(numpy=autograd.numpy))
    return predict(*args, **kwargs)


def objective_f(params, batch_input, batch_output, dropout=False):
    layer = predict_f(params, batch_input, dropout=dropout)
    W, h = params[-1]
    z = numpy.dot(layer, W) + h  # no relu on last one
    log_p = z - logsumexp(z, axis=1, keepdims=True)
    goal = numpy.sum(log_p * batch_output)
    return -goal


f = h5py.File('colors.hdf5')
rgb = numpy.array(f['rgb'])
labels = numpy.array(f['labels'])

# Generate params
n_output = max(labels) + 1
layers = [3, 64, 64, 64, 64, 16, n_output]
if os.path.exists(PARAMS_PICKLED):
    with open(PARAMS_PICKLED, 'rb') as f:
        params = pickle.load(f)
else:
    params = []
    for i in range(1, len(layers)):
        params.append((numpy.random.randn(layers[i-1], layers[i]) * 1e-1,
                       numpy.random.randn(layers[i]) * 1e-1))

# Derivative
objective_grad = grad(objective_f)

# Maximize log-likelihood
step_size = 1e-2
last_avg_objective = float('inf')
for epoch in range(100):
    indexes = list(range(rgb.shape[0]))
    random.shuffle(indexes)
    batch_size = 2**10
    terms = []
    ns = []

    def get_data(i):
        offset_start = i*batch_size
        offset_end = min((i+1)*batch_size, rgb.shape[0])
        batch_indexes = indexes[offset_start:offset_end]
        batch_input = numpy.array([rgb[i]/255 for i in batch_indexes])
        batch_output = numpy.zeros((offset_end - offset_start, n_output))
        for i, j in enumerate(batch_indexes):
            batch_output[i,labels[j]] = 1
        return batch_input, batch_output

    def callback(params, i, g):
        batch_input, batch_output = get_data(i)
        terms.append(objective_f(params, batch_input, batch_output))
        ns.append(batch_input.shape[0])
        print('%3d %6.2f%% %9.6f %9.2f' % (epoch, 100 * i / rgb.shape[0] * batch_size, step_size, numpy.exp(sum(terms) / sum(ns))))

    def grad_i(params, i):
        batch_input, batch_output = get_data(i)
        return objective_grad(params, batch_input, batch_output)

    params = adam(grad_i, params, step_size=step_size, num_iters=int(numpy.ceil(rgb.shape[0]/batch_size)), callback=callback)

    if numpy.exp(sum(terms) / sum(ns)) > last_avg_objective * (1 - 1e-3):
        step_size *= 0.1
        print('learning rate now', step_size)
    last_avg_objective = numpy.exp(sum(terms) / sum(ns))
    if step_size < 1e-7:
        break
    with open(PARAMS_PICKLED, 'wb') as f:
        pickle.dump(params, f)
