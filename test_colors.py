import numpy
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie1994, delta_e_cie2000
from colormath.color_conversions import convert_color
from chromaticity.conversion import rgb2lab, cie94, ciede2000


def test_rgb2lab():
    expected = numpy.array([[  53.24,  80.09,   67.20],
                            [  87.73, -86.18,   83.18],
                            [  32.30,  79.19, -107.86]])
    actual = rgb2lab(numpy.eye(3))
    assert numpy.max(abs(expected - actual)) < 1e-2


def test_cie94():
    for p in numpy.eye(3):
        for q in numpy.eye(3):
            cp = convert_color(sRGBColor(*p), LabColor)
            cq = convert_color(sRGBColor(*q), LabColor)
            assert abs(delta_e_cie1994(cp, cq) - cie94(rgb2lab(p), rgb2lab(q))) < 1e-2


def test_ciede2000():
    for p in numpy.eye(3):
        for q in numpy.eye(3):
            cp = convert_color(sRGBColor(*p), LabColor)
            cq = convert_color(sRGBColor(*q), LabColor)
            print(p, q, delta_e_cie2000(cp, cq), ciede2000(rgb2lab(p), rgb2lab(q)))
