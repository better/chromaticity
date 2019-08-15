import numpy
import pickle
import PIL.Image, PIL.ImageDraw
from nn import predict
from colors import rgb2lab, cie94, ciede2000


class RGB:
    def embed(self, colors):
        return colors

    def dist(self, p, q):
        return numpy.sum((last_vector - all_embedded)**2, axis=1)


class Lab:
    def embed(self, colors):
        return rgb2lab(colors)

    def dist(self, p, q):
        return numpy.sum((last_vector - all_embedded)**2, axis=1)


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
        with open('nn.pickle', 'rb') as f:
            self.params = pickle.load(f)

    def embed(self, colors):
        embedded = predict(self.params, colors, dropout=False)
        return embedded / numpy.sum(embedded**2, axis=1)[:,numpy.newaxis]

    def dist(self, p, q):
        return numpy.sum((last_vector - all_embedded)**2, axis=1)


if __name__ == '__main__':
    # Build matrix of 16x16x16 colors
    vs = list(range(0, 256, 17))
    all_rgb = []
    for r in vs:
        for g in vs:
            for b in vs:
                all_rgb.append((r/255, g/255, b/255))
    all_rgb = numpy.array(all_rgb)

    for fn, obj in [
            ('0_rgb.png', RGB()),
            ('1_lab.png', Lab()),
            ('2_cie94.png', CIE94()),
            ('3_ciede2000.png', CIEDE2000()),
            ('4_nn.png', NN())
            ]:
        all_embedded = obj.embed(all_rgb)
        colors = [0]  # , len(all_rgb)-1]
        min_dist = numpy.array([float('inf')]*len(all_rgb))

        for i in range(16**2):
            if len(colors) <= i:
                # Find the next color
                j = numpy.argmax(min_dist)
                print(j, numpy.max(min_dist))
                colors.append(j)

            # Update min dist for this color
            last_vector = all_embedded[colors[i]][numpy.newaxis,:]
            min_dist = numpy.minimum(min_dist, obj.dist(last_vector, all_embedded))  # numpy.sum((last_vector - all_embedded)**2, axis=1))

        d, e = 40, 10
        k = int(numpy.ceil(len(colors)**0.5))
        im = PIL.Image.new('RGB', (2*k*d+3*e, k*d+2*e))
        draw = PIL.ImageDraw.Draw(im)

        def draw_patches(colors, x_offset, y_offset):
            for i, color in enumerate(colors):
                x, y = i%k, i//k
                draw.rectangle((x*d+x_offset, y*d+y_offset, (x+1)*d+x_offset, (y+1)*d+y_offset), fill=tuple(int(255*z) for z in all_rgb[color]))

        draw_patches(colors, e, e)
        colors = sorted(colors, key=lambda j: all_rgb[j][1] - all_rgb[j][0])
        for o1 in range(0, len(colors), k):
            o2 = min(o1+k, len(colors))
            colors[o1:o2] = sorted(colors[o1:o2], key=lambda j: all_rgb[j][2] - all_rgb[j][1])

        draw_patches(colors, k*d+2*e, e)
        im.save(fn)

    
