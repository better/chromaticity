import numpy
import pickle
import PIL.Image, PIL.ImageDraw
from nn import predict
from colors import rgb2lab, cie94, ciede2000


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
        with open('nn.pickle', 'rb') as f:
            self.params = pickle.load(f)

    def embed(self, colors):
        embedded = predict(self.params, colors, dropout=False)
        return embedded / numpy.sum(embedded**2, axis=1)[:,numpy.newaxis]

    def dist(self, p, q):
        return numpy.sum((p - q)**2, axis=1)


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
        all_rgb_embedded = obj.embed(all_rgb)
        colors = [numpy.array(z)/0xff for z in [[0x29, 0x18, 0x42], [0x37, 0xeb, 0xc2], [0x00, 0xc3, 0x97], [0xdf, 0x16, 0x83], [0x42, 0x12, 0x87]]]
        colors_embedded = obj.embed(numpy.array(colors))

        min_dist = numpy.array([float('inf')]*len(all_rgb))

        for i in range(16**2):
            if len(colors) <= i:
                # Find the next color
                j = numpy.argmax(min_dist)
                print(j, numpy.max(min_dist))
                colors.append(all_rgb[j])
                last_vector_embedded = all_rgb_embedded[j]
            else:
                last_vector_embedded = colors_embedded[i]

            # Update min dist for this color
            last_vector_embedded = last_vector_embedded[numpy.newaxis,:]
            min_dist = numpy.minimum(min_dist, obj.dist(last_vector_embedded, all_rgb_embedded))

        d, e = 40, 10
        k = int(numpy.ceil(len(colors)**0.5))
        im = PIL.Image.new('RGB', (2*k*d+3*e, k*d+2*e))
        draw = PIL.ImageDraw.Draw(im)

        def draw_patches(colors, x_offset, y_offset):
            for i, color in enumerate(colors):
                x, y = i%k, i//k
                draw.rectangle((x*d+x_offset, y*d+y_offset, (x+1)*d+x_offset, (y+1)*d+y_offset), fill=tuple(int(255*z) for z in color))

        draw_patches(colors, e, e)
        colors = sorted(colors, key=lambda c: c[1] - c[0])
        for o1 in range(0, len(colors), k):
            o2 = min(o1+k, len(colors))
            colors[o1:o2] = sorted(colors[o1:o2], key=lambda c: c[2] - c[1])

        draw_patches(colors, k*d+2*e, e)
        im.save(fn)

    
