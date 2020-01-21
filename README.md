This is an experimental package to generate color palettes. We train a neural network to generate a color space that's _perceptually uniform_. Older classic methods do not have this property.

RGB (most simple method)

![pic](pics/rgb.png)

Euclidean distance in CIELAB color space:

![pic](pics/lab.png)

CIE94 color difference formula:

![pic](pics/cie94.png)

CIEDE2000 color difference formula:

![pic](pics/ciede2000.png)

Our method (neural network based):

![pic](pics/nn.png)
