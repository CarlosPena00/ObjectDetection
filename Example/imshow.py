#http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt

# FromFile display image
# filename name of image file
# fmt formate of the image
# Return image shape as list

def fromFile(filename,fmt='.jpg'):
    img = sp.misc.imread(filename + fmt) 
    img_tf = tf.Variable(img)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    im = sess.run(img_tf)
    plt.imshow(im)
    plt.show()
    return img_tf.get_shape().as_list() 

# FromImg display image
# img img src
# Return image shape as list
#NOT READY
def fromImg(img):
    img_tf = tf.Variable(img)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    im = sess.run(img_tf)
    plt.imshow(im)
    plt.show()
    return img_tf.get_shape().as_list() 
