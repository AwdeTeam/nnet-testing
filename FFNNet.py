print("Running imports...")
import numpy
import sklearn
print("Finished imports.")

class FeedForwardNeuralNetwork():

	#member variables
	shape = [] 	#size - 2 is the number of hidden variables, 
				#shape[0] is number of inputs, shape[n] is number of neurons in layer n
				
	weights = [][] 	#if layer n has k neurons, the weights for that neuron are 
					#weights[n][0] through weights[n][k-1] and weights[k] is the bias
					
	initialized = false
	
	def _init_(self, shape):
		initialized = true
	
	def load(self, weights):
		self.weights = weights
		initialized = true
		for i in range (0, len(weights)):
			self.shape.append(len(weights[i]) - 1)
	
	
