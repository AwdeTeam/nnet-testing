print "(Running imports...)"
import theano 
import theano.tensor as t
import numpy
print "(Imports complete!)"

# NOTE: For now, weights will be randomly initialized

class NeuralNetwork():
	
	# member variables
	trainingInputs = []
	trainingOutputs = []
	weights = []

	inputs = 0
	layers = 0
	layerNueronCount = 0
	outputs = 0
	

	# construction
	def __init__(self,inpNum,hiddenLayerNum,hiddenLayerNeurons,outNum):
		print "Network shape initialized: " + str(inpNum) + " inputs, " + str(hiddenLayerNum) + " hidden layers, " + str(hiddenLayerNeurons) + " neurons per hidden layer, " + str(outNum) + " outputs"
		self.inputs = inpNum
		self.layers = hiddenLayerNum
		self.outputs = outNum
		

	# setup and compile the necessary theano functions
	def initTheanoFunctions(self):
		print "(Compiling theano functions...)"

		mat_incoming = t.dmatrix('mat_incoming') # layer inputs
		mat_weights = t.dmatrix('mat_weights') # layer connection weights
		mat_presig = t.dot(mat_incoming, mat_weights) # dot product of inputs with weights (no sigmoid yet)

		mat_outputs = 1 / (1 + t.exp(-mat_presig)) # apply sigmoid

		self.feedForward = theano.function([mat_incoming, mat_weights], mat_outputs)

		print "(Functions ready!)"


	# expects csv file
	def readTrainingData(self, fileName):
		lines = [line.strip() for line in open(fileName)]
		for line in lines:
			csv = line.split(",")

			# get training inputs
			inputArray = numpy.asarray([[]])
			for i in range(0,self.inputs):
				inputArray = numpy.concatenate((inputArray, [[csv[i]]]), 1) # 1 specifies axis (adds col, not row)
			self.trainingInputs.append(inputArray)

			# get training outputs
			outputArray = numpy.asarray([[]])
			for i in range(0 + self.inputs, self.inputs + self.outputs):
				outputArray = numpy.concatenate((outputArray, [[csv[i]]]), 1)
			self.trainingOutputs.append(outputArray)	
			
		# print self.trainingInputs
		# print self.trainingOutputs

	def generateWeights(self):
		for i in range(0,self.layers + 1):
			
	
