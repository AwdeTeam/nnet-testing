print("Running imports...")
import numpy
import sklearn
#import metrics from sklearn
print("Finished imports.")

class SupervisedClassifier():
	
	#member variables
	A = None #root algorithm
	ID = "FF-00" #no algorithm set
	
	def _init_(self, id):
		if("naive_bayes" or id == "00-00"):
			self.A = GaussianNB()
			self.ID = "00-00"
		elif(id == "feed_forward" or id == "00-01"):
			self.A = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1) #gonna need a way to pass in arguments at some point
			self.ID = "00-01"
	
	def predict(self, set):
		return self.A.predict(set)

		
	def fit(self, set, exp, ratio):
		point = int((1-ratio)*len(exp))
		target = exp[:point]
		tmp = self.A.fit(set[:point], target)
		
		return metrics.accuracy_score(exp[point:], tmp.predict[set:], normalize=True, sample_weight=None)
	
	def fit(self, set, exp):
		return fit(self, set, exp, 0.1) #reserve 10% of the dataset for accuracy test by default
		
	def metafit(self, I, set, exp, ratio): #yes, I know this could be handled upstream, but I think it fits better here
		return fit(self, I.predict(I, set), exp, ratio)
		
	def metafit(self, I, set, exp):
		return fit(self, I.predict(I, set), exp)
		
		
class Dataset():

	#member variables
	rawData = None #matrix of strings loaded directly from a dataset, top row is categories
	normalData = None #normalized to passed specifications and uniform type (default to reals (internally 32-bit floating points) bounded by [-1, 1])
	categories = [] #keep track of the name of the categories
	type = "real"
	bound = [-1, 1]
	
	def loadFromText(self, file, delim):
		self.rawData = numpy.loadtxt(open(file,"rb"), delimiter=delim, skiprows=0, ndimin=2)
		for t in range (0, colsInRawData)
			categories[t] = categories[0][t]
			
		for i in range (1, rowsInRawData):
			for j in range (0, colsInRawData):
				normalData[i][j] = rawData[i][j]
		
	def 
		
print("nothing broken")