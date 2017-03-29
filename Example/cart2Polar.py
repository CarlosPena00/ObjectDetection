#http://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy


import numpy
import scipy

# Cart2Polar cartesian to Polar
# String file name 
# fmt format type, like .jpg .jpeg ...
# normalize  bool 
#absAngle 1 to [0,180] ; 0 to [-180,180]
def cart2Polar(string,fmt = '.jpg', normalize=0, absAngle = 1):
    im = scipy.misc.imread(string+fmt)
    im = im.astype('int32')
    dx = scipy.ndimage.sobel(im, 0) 
    dy = scipy.ndimage.sobel(im, 1)  
    mag = numpy.hypot(dx, dy)  # sqrt(dx*dx + dy*dy)
    teta = numpy.rad2deg(numpy.arctan2(dy,dx)) #arctan(dy/dx) OBS:[-180, 180]
    if normalize:
        mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
    if absAngle:
        teta = numpy.abs(teta)
    scipy.misc.imsave(string+'_mag'+fmt, mag)
    scipy.misc.imsave(string+'_ang'+fmt, teta)
    scipy.misc.imsave(string+'_dx'+fmt, dx)
    scipy.misc.imsave(string+'_dy'+fmt, dy)
    return mag,teta


mag , teta = cart2Polar ("test")
