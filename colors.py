import numpy
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie1994, delta_e_cie2000
from colormath.color_conversions import convert_color


def lab_f(t):
    delta = 6/29
    return (t > delta**3) * t**(1/3) + (t <= delta**3) * (t / (3*delta**2) + 4/29)


def rgb2lab(x):
    # numpy-vectorized version
    matrix = numpy.array([[0.412453, 0.357580, 0.180423],
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]])
    factor = numpy.array([0.950456, 1, 1.088754])
    cie = numpy.dot(x, matrix.T) / factor
    
    matrix2 = numpy.array([[   0,  500,    0],
                           [ 116, -500,  200],
                           [   0,    0, -200]])
    Lab = numpy.dot(lab_f(cie), numpy.array(matrix2)) + numpy.array([-16, 0, 0])
    return Lab


# TODO: make this into a unit test
expected = numpy.array([[  53.24,  80.09,   67.20],
                        [  87.73, -86.18,   83.18],
                        [  32.30,  79.19, -107.86]])
actual = rgb2lab(numpy.eye(3))
assert numpy.max(abs(expected - actual)) < 1e-2

def cie94(p, q):
    L_diff = numpy.dot(p - q, numpy.array([1, 0, 0]))
    Cp = numpy.dot(p**2, numpy.array([0, 1, 1]))**0.5
    Cq = numpy.dot(q**2, numpy.array([0, 1, 1]))**0.5
    C_diff = Cp - Cq
    a_diff = numpy.dot(p - q, numpy.array([0, 1, 0]))
    b_diff = numpy.dot(p - q, numpy.array([0, 0, 1]))
    H_diff = numpy.maximum(a_diff**2 + b_diff**2 - C_diff**2, 0)**0.5
    KL, KC, KH, K1, K2 = (1, 1, 1, 0.045, 0.015)
    SL, SC, SH = 1, 1+K1*Cp, 1+K2*Cp
    return ((L_diff / (KL*SL))**2 +
            (C_diff / (KC*SC))**2 +
            (H_diff / (KH*SH))**2)**0.5


for p in numpy.eye(3):
    for q in numpy.eye(3):
        cp = convert_color(sRGBColor(*p), LabColor)
        cq = convert_color(sRGBColor(*q), LabColor)
        assert abs(delta_e_cie1994(cp, cq) - cie94(rgb2lab(p), rgb2lab(q))) < 1e-2


def ciede2000(p, q):
    L1, a1, b1 = numpy.moveaxis(p, -1, 0)
    L2, a2, b2 = numpy.moveaxis(q, -1, 0)
    delta_L_prime = L1 - L2
    L_bar = (L1 + L2)/2
    C1 = (a1**2 + b1**2)**0.5
    C2 = (a2**2 + b2**2)**0.5
    C_bar = (C1 + C2)/2
    f = 1 - (C_bar**7 / (C_bar**7 + 25**7))**0.5
    a1_prime = a1 * (1 + f/2)
    a2_prime = a2 * (1 + f/2)
    C1_prime = (a1_prime**2 + b1**2)**0.5
    C2_prime = (a2_prime**2 + b2**2)**0.5
    delta_C_prime = C2_prime - C1_prime
    C_bar_prime = (C1_prime + C2_prime)/2
    h1_prime = numpy.arctan2(b1, a1_prime)
    h2_prime = numpy.arctan2(b2, a2_prime)
    h1_prime = h1_prime + (h1_prime < 0) * 2 * numpy.pi
    h2_prime = h2_prime + (h2_prime < 0) * 2 * numpy.pi
    # delta_h_prime = numpy.fmod(h2_prime - h1_prime + 3*numpy.pi, 2*numpy.pi) - numpy.pi
    delta_h_prime = h2_prime - h1_prime
    delta_h_prime = delta_h_prime + (numpy.fabs(delta_h_prime) > numpy.pi) * 2*numpy.pi - (h2_prime > h1_prime) * 4*numpy.pi
    delta_H_prime = 2*(C1_prime*C2_prime)**0.5 * numpy.sin(delta_h_prime/2)
    # H_bar_prime = (numpy.fmod(h1_prime + h2_prime + numpy.pi, 2*numpy.pi) - numpy.pi)/2
    H_bar_prime = (((numpy.fabs(h1_prime - h2_prime) > numpy.pi) * 2*numpy.pi) + h1_prime + h2_prime) / 2.0
    T = 1 - 0.17 * numpy.cos(H_bar_prime - numpy.pi/6) \
        + 0.24*numpy.cos(2*H_bar_prime) \
        + 0.32*numpy.cos(3*H_bar_prime + numpy.pi/30) \
        - 0.20*numpy.cos(4*H_bar_prime - 2*numpy.pi*63/360)
    SL = 1 + 0.015 * (L_bar - 50)**2 / (20 + (L_bar - 50)**2)**0.5
    SC = 1 + 0.045 * C_bar_prime
    SH = 1 + 0.015 * C_bar_prime*T
    RT = -2*(C_bar_prime**7 / (C_bar_prime**7 + 25**7))**0.5 * numpy.sin(numpy.pi/3 * numpy.exp(-((H_bar_prime - 2*numpy.pi*275/360)/(2*numpy.pi*25/360))**2))
    KL, KC, KH = 1, 1, 1
    return ((delta_L_prime / (KL * SL))**2 +
            (delta_C_prime / (KC * SC))**2 +
            (delta_H_prime / (KH * SH))**2 +
            RT * delta_C_prime * delta_H_prime / (KC * SC * KH * SH))**0.5


for p in numpy.eye(3):
    for q in numpy.eye(3):
        cp = convert_color(sRGBColor(*p), LabColor)
        cq = convert_color(sRGBColor(*q), LabColor)
        print(p, q, delta_e_cie2000(cp, cq), ciede2000(rgb2lab(p), rgb2lab(q)))
