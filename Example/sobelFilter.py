#http://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy


import numpy
import scipy
from scipy import ndimage

im = scipy.misc.imread('test.jpg')
im = im.astype('int32')
dx = ndimage.sobel(im, 0)  # horizontal derivative
dy = ndimage.sobel(im, 1)  # vertical derivative
mag = numpy.hypot(dx, dy)  # magnitude
mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
scipy.misc.imsave('sobel.jpg', mag)

