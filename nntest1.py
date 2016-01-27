print "Importing stuffs..."

import theano
import theano.tensor as t
import numpy
import numpy.random as rng # random number generator

#rng.permutation(3) # random generator for hidden layer values

# ----- theano stuff! -----

print "Compiling stuffs..."

# first theano matrices
inputs = t.dmatrix('inputs') # inputs
weights1 = t.dmatrix('weights1') # first layer weights
out1 = t.dot(inputs, weights1) # hidden layer outputs

# first layer function
f1 = theano.function([inputs,weights1], out1)

# second layer weights
weights2 = t.dmatrix('weights2')

# end result outputs
outputs = t.dot(out1, weights2)

# end result calculator function
f2 = theano.function([out1,weights2], outputs)



# ----- execution stuff! -----

print "Executing stuffs!"

# matricies!
mat_input = numpy.asarray([[10,5]])
mat_weights1 = numpy.asarray([
	[5,3,7],
	[4,5,4]])
mat_weights2 = numpy.asarray([
	[-2],
	[7],
	[-5]])

mat_hiddenLayer = f1(mat_input,mat_weights1)
finalLayer = f2(mat_hiddenLayer, mat_weights2)
print finalLayer
