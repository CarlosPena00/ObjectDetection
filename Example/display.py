#http://stackoverflow.com/questions/33648322/tensorflow-image-reading-display
from scipy import misc
import tensorflow as tf
img = misc.imread('test.jpg') 
img_tf = tf.Variable(img)
#print img_tf.get_shape().as_list() 

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
im = sess.run(img_tf)


import matplotlib.pyplot as plt
fig = plt.figure()
#fig.add_subplot(1,2,1)
plt.imshow(im)
#fig.add_subplot(1,2,2)
#plt.imshow(img)
plt.show()

