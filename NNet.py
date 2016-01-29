print "(Running imports...)"
import theano 
import theano.tensor as t
import numpy
print "(Imports complete!)"

# NOTE: For now, weights will be randomly initialized

class NeuralNetwork():
	
	# member variables
	

	# construction
	def __init__(self,inpNum,hiddenLayerNum,outNum):
		print "Being initialized: " + str(inpNum) + " inputs " + str(hiddenLayerNum) + " hidden layers " + str(outNum) + " outputs"
		

	# setup and compile the necessary theano functions
	def initTheanoFunctions(self):
		print "(Compiling theano functions...)"

		mat_incoming = t.dmatrix('mat_incoming') # layer inputs
		mat_weights = t.dmatrix('mat_weights') # layer connection weights
		mat_presig = t.dot(mat_incoming, mat_weights) # dot product of inputs with weights (no sigmoid yet)

		mat_outputs = 1 / (1 + t.exp(-mat_presig)) # apply sigmoid

		self.feedForward = theano.function([mat_incoming, mat_weights], mat_outputs)

		print "(Functions ready!)"

	def readTrainingData(self, fileName):
		lines = [line.strip() for line in open(fileName)]
		for line in lines:
			print "thing: " + line
