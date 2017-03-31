#http://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy


import numpy
import scipy as sp
import imshow
from  scipy import ndimage

# Cart2Polar cartesian to Polar
# filename name of image file 
# fmt format type, like .jpg .jpeg ...
# normalize  bool 
#absAngle 1 to [0,180] ; 0 to [-180,180]
def cart2PolarFile(filename,fmt = '.jpg', normalize=0, absAngle = 1):
    im = sp.misc.imread(filename+fmt)
    im = im.astype('int32')
    dx = ndimage.sobel(im, 0) 
    dy = ndimage.sobel(im, 1)  
    mag = numpy.hypot(dx, dy)  # sqrt(dx*dx + dy*dy)
    teta = numpy.rad2deg(numpy.arctan2(dy,dx)) #arctan(dy/dx) OBS:[-180, 180]
    if normalize:
        mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
    if absAngle:
        teta = numpy.abs(teta)
    sp.misc.imsave(filename+'_mag'+fmt, mag)
    sp.misc.imsave(filename+'_ang'+fmt, teta)
    sp.misc.imsave(filename+'_dx'+fmt, dx)
    sp.misc.imsave(filename+'_dy'+fmt, dy)
    return mag,teta

#NOT READY
def cart2Polar(im, normalize=0, absAngle = 1):
    im = im.astype('int32')
    dx = ndimage.sobel(im, 0) 
    dy = ndimage.sobel(im, 1)  
    mag = numpy.hypot(dx, dy)  # sqrt(dx*dx + dy*dy)
    teta = numpy.rad2deg(numpy.arctan2(dy,dx)) #arctan(dy/dx) OBS:[-180, 180]
    if normalize:
        mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
    if absAngle:
        teta = numpy.abs(teta)
    return mag,teta


