import numpy
import os
import PIL.Image, PIL.ImageDraw
from colors.palettes import RGB, Lab, CIE94, CIEDE2000, NN, Cycler


DIR = 'pics'
if not os.path.exists(DIR):
    os.makedirs(DIR)

for fn, obj in [
    ('rgb.png', RGB),
    ('lab.png', Lab),
    ('cie94.png', CIE94),
    ('ciede2000.png', CIEDE2000),
    ('nn.png', NN)
    ]:
    cycler = Cycler(method=obj)
    colors = cycler[:2**8]

    # Generate Matplotlib colormap format
    # print([list(c) + [1] for c in colors])

    d, e = 40, 10
    k = int(numpy.ceil(len(colors)**0.5/2))*2
    colors = numpy.reshape(colors, (k, k, 3))
    im = PIL.Image.new('RGB', (2*k*d+3*e, k*d+2*e))
    draw = PIL.ImageDraw.Draw(im)

    def draw_patches(colors, x_offset, y_offset):
        for row in range(colors.shape[0]):
            for col in range(colors.shape[1]):
                draw.rectangle((col*d+x_offset, row*d+y_offset, (col+1)*d+x_offset, (row+1)*d+y_offset), fill=tuple(int(255*z) for z in colors[row][col]))

    draw_patches(colors, e, e)
    colors = numpy.array(sorted(numpy.reshape(colors, (k*k, 3)), key=lambda c: sum(c)))
    colors = numpy.reshape(colors, (2, 2, k//2, k//2, 3))
    for i in range(2):
        for j in range(2):
            rect = colors[i,j]
            for _ in range(100):
                for row in range(rect.shape[1]):
                    rect[row,:] = numpy.array(sorted(rect[row,:], key=lambda c: c[1] - c[0]))
                for col in range(rect.shape[2]):
                    rect[:,col] = numpy.array(sorted(rect[:,col], key=lambda c: c[2] - c[1]))
            draw_patches(rect, k*d+2*e + i*k/2*d, e + j*k/2*d)

    im.save(os.path.join(DIR, fn))
