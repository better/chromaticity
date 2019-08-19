import numpy
import os
import pickle
from colors.nn import predict, PARAMS_PICKLED
from colors.conversion import rgb2lab, cie94, ciede2000


class RGB:
    def embed(self, colors):
        return colors

    def dist(self, p, q):
        return numpy.sum((p - q)**2, axis=1)


class Lab:
    def embed(self, colors):
        return rgb2lab(colors)

    def dist(self, p, q):
        return numpy.sum((p - q)**2, axis=1)


class CIE94:
    def embed(self, colors):
        return rgb2lab(colors)

    def dist(self, p, q):
        return cie94(p, q)


class CIEDE2000:
    def embed(self, colors):
        return rgb2lab(colors)

    def dist(self, p, q):
        return ciede2000(p, q)


class NN:
    def __init__(self):
        with open(PARAMS_PICKLED, 'rb') as f:
            self.params = pickle.load(f)

    def embed(self, colors):
        embedded = predict(self.params, colors, dropout=False)
        return embedded / numpy.sum(embedded**2, axis=1)[:,numpy.newaxis]**0.5

    def dist(self, p, q):
        return numpy.sum((p - q)**2, axis=1)


class Cycler:
    def __init__(self, method=NN, remove_bw=True, seed=None, eps=1e-9):
        # Build matrix of 16x16x16 colors
        vs = list(range(0, 256, 17))
        all_rgb = []
        for r in vs:
            for g in vs:
                for b in vs:
                    all_rgb.append((r/255, g/255, b/255))
        self.all_rgb = numpy.array(all_rgb)
        self.m = method()
        self.all_rgb_embedded = self.m.embed(self.all_rgb)
        if seed is None:
            seed = numpy.eye(3)
        else:
            seed = numpy.array(seed)
        if remove_bw:
            seed = numpy.vstack([numpy.array([(0, 0, 0), (1, 1, 1)]), seed])
        self.seed = seed
        self.seed_embedded = self.m.embed(seed)
        self.colors = []
        self.min_dist = numpy.array([float('inf')]*len(self.all_rgb))
        self.eps = eps
        self.remove_bw = remove_bw

    def __iter__(self):
        def gen():
            i = 0
            while True:
                yield self[i]
                i += 1
        return gen()

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(2**99))]
        assert isinstance(key, int)
        if self.remove_bw:
            key += 2
        while len(self.colors) <= key:
            if key < len(self.seed_embedded):
                # Just use the seed
                j = len(self.colors)
                color = self.seed[j]
                last_vector_embedded = self.seed_embedded[j]
            else:
                # Find the next color
                j = numpy.argmax(self.min_dist)
                color = self.all_rgb[j]
                last_vector_embedded = self.all_rgb_embedded[j]

            # Update min dist for this color
            last_vector_embedded = last_vector_embedded[numpy.newaxis,:]
            self.min_dist = numpy.minimum(self.min_dist, self.m.dist(last_vector_embedded, self.all_rgb_embedded))
            self.min_dist += self.eps
            self.colors.append(color)

        return self.colors[key]
