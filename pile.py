from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import itertools

if tf.__version__ < '2.0.0':
    raise ImportError(
        "Need TF 2.0.0 at least but you have {}".format(tf.__version__))

# Image width and height.
# Grains that fall off the edge are removed.
# Bear in mind this will introduce edge effects if your
# image isn't large enough.
# Use odd sizes for symmetry if edge effects are involved.
# (GPUs may prefer powers of two though.)
h = 511
w = 511

# The data type used for computation.
# I'd rather use uint32 but tf only supports floating point
# types for convolution.
float_type = np.float32

# The size of the initial pile is `base_size * 2 ** doublings`
# If doublings == 0 then the sequence of images gives a nice
# history of the reduction process.
#
# If doublings > 0 it's using a "nonlinear" reduction strategy
# which doesn't look as good at intermediate stages but may
# go a little faster. (Same result in the end.)
# Eg. `base_size == 1`, `doublings == 14`
# should give same result as
# Eg. `base_size == 2 ** 14`, `doublings == 0`.
#
# Don't allow `4 * base_size` to be larger than the largest
# number representable in the float type used.
# (Set `init_background` to zero if `doublings > 0`.)
base_size = 2 ** 14
doublings = 0

# A pixel colour is the number of sand grains in the cell
# muliplied by this colour modulo 256
colour = np.array([102, 182, 65])

# We can pre-seed the background with grains
# Note that this gets multiplied by the doublings too
# So expect veeeerrrrrrrry slow progress if this is non-zero
# and you use doublings.
# (Probably this should be < 4, but you can try >= 4 if you like.)
init_background = 0

# Number of reduction steps between each image output
iterations_per_image = 1024

# Output file name is out.XXXX.YYYY.jpg
# where XXXX is the current doubling and
#       YYYY is the iteration on the current doubling

# Other things to try:
# The `scatter_nd` line below scatters the initial sand grains.
# It's easily modified.
# The first argument to `scatter_nd` is the list of coordinates
# and the second argument is the list of corresponding numbers of grains.
#
# You can try varying the kernel, eg. for a hexagonal grid
# try [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
# and warping the resulting image.

background = tf.fill((h, w), float_type(init_background))
init = background + tf.scatter_nd([[h//2, w//2]],
                                  [float_type(base_size)], (h, w))
init = init[None, :, :, None]

kernel = np.array([
  [0, 1, 0],
  [1, 0, 1],
  [0, 1, 0]], dtype=float_type)
kernel = kernel[:, :, None, None]
neighbours = np.sum(kernel)

pile = init

@tf.function
def reduce(pile):
  remainders = tf.math.floormod(pile, neighbours)
  topples = tf.math.floordiv(pile, neighbours)
  collected = tf.nn.conv2d(topples, kernel,
                           strides=[1, 1, 1, 1],
                           padding="SAME")
  return remainders + collected

@tf.function
def is_stable(pile):
  return tf.math.reduce_all(pile < neighbours)

@tf.function
def make_rgb(pile):
  image = pile[0, :, :, 0]
  return tf.cast(tf.math.floormod(colour[None, None, :] * image[:, :, None],
                                  256.),
                 dtype=np.uint8)

for power in range(doublings + 1):
  for t in itertools.count():
    for i in xrange(iterations_per_image):
      # Stability checking is expensive. It might be worth
      # doing more reductions before each check.
      if is_stable(pile):
          break
      pile = reduce(pile)

    image = make_rgb(pile)
    filename = 'out.{:04d}.{:04d}.jpg'.format(power, t)
    tf.io.write_file(filename, tf.io.encode_jpeg(image))
    print("Wrote to '{}'".format(filename))

    if is_stable(pile):
        break

  pile = pile * 2.0
