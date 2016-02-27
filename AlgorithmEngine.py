print("(Running imports...)")
import math
import csv
import numpy
import sklearn
from sklearn.naive_bayes import GaussianNB
import joblib #THIS COULD BE VERY INSECURE, CHECK WITH SOMEONE WHO ACTUALLY KNOWS WHAT THEY'RE TALKING ABOUT!
#import metrics from sklearn
print("(Finished imports.)")

class Sharable():
	#member variables
	algorithm = None #the trained algorithm object
	name = ""
	version = ""
	statistics = None #this will probably be an sklearn helper class. We might want to wrap it up with other stuff also
	computation = None #we might have to implement this in numpy
	client_history = None #client-side provenance
	usage = None #input and output types, along with usage intentions
	
	def __init__(self, algorithm, name, version, statistics, computation, client_history, usage):
		self.algorithm = algorithm
		self.name = name
		self.version = version
		self.statistics = statistics
		self.client_history = client_history
		self.usage = usage
		
	def saveState(self, path):
		joblib.dump(self, path + name + "_" + version + ".saf", compress=2, cache_size=100, protocol=None)
		
	def loadState(file): #verify upstream that this is a .saf, not a .paf
		return joblib.load(file)
		
	def exportPAF(self, path):
		#some sort of hashing thing to allow the ability to verify integrity
		joblib.dump(self, path + name + "_" + version + ".paf", compress=9, cache_size=100, protocol=None)

	def importPAF(self, path):
		#some sort of hash verification to ensure this is the expected thing
		return joblib.load(file)

class SupervisedClassifier():
	
	#member variables
	A = None #root algorithm
	ID = "FF-00" #no algorithm set
	
	def __init__(self, id): #right... *two* underscores
		if("naive_bayes" or id == "00-00"):
			self.A = GaussianNB()
			self.ID = "00-00"
		elif(id == "feed_forward" or id == "00-01"):
			self.A = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1) #gonna need a way to pass in arguments at some point
			self.ID = "00-01"
	
	def predict(self, set):
		return self.A.predict(set)

		
	def fit(self, set, exp, ratio=0.1):
		point = int((1-ratio)*len(exp))
		target = exp[:point]
		tmp = self.A.fit(set[:point], target)
		
		return metrics.accuracy_score(exp[point:], tmp.predict[set:], normalize=True, sample_weight=None)
	
	#def fit(self, set, exp):
	#	return fit(self, set, exp, 0.1) #reserve 10% of the dataset for accuracy test by default
		
	def metafit(self, I, set, exp, ratio=0.1): #yes, I know this could be handled upstream, but I think it fits better here
		return fit(self, I.predict(I, set), exp, ratio)
		
	#def metafit(self, I, set, exp):
	#	return fit(self, I.predict(I, set), exp)
		
class Datatype():
	# can be Null, Boolean, Integer, BoundedReal, Real, Complex, String, Image, or Aggregate
    name = "unnamed";
    rank = -1;
    #private Datatype[] bundle = null;
		
class Dataset():

	#member variables
	rawData = None #matrix of strings loaded directly from a dataset, top row is categories
	normalData = None #normalized to passed specifications and uniform type (default to reals (internally 32-bit floating points) bounded by [-1, 1])
	datatype = None #datatype of the set TODO make this work

	dataRows = 0
	dataCols = 0
	
	categories = [] #keep track of the name of the categories
	#type = "real"
	#bound = [-1, 1] # NOTE: based on current method, not using hard limits of -1 and 1, rather using -mean / std method, which seems to be fairly conventional
	
	def indexColumn(self, dex, normalized=False):
		return self.getRawData[:, dex]
	
	def getColumn(self, name, normalized=False):
		dex = -1
		#print(self.getRawData)
		for i in range(0, len(self.categories)):
			if(name == self.categories[i]):
				dex = i
				break
		if(dex == -1):
			return None
		
		return self.getNormalData()[:,dex] if normalized else self.getRawData()[:,dex]
	
	def excludeColumn(self, name, normalized=False):
		dex = -1
		ref = self.getNormalData() if normalized else self.getRawData()
		print(str(type(ref)) + "\n" + str(ref) + "\nRows: " + str(len(ref)) + "\tCols: " + str(len(ref[0])))
		r = numpy.array([len(ref) - 1, len(ref[0])])
		for i in range(0, len(self.categories)):
			if(name != self.categories[i]):
				dex = i
		ref = numpy.delete(ref, (dex), axis=1)
		print(str(type(ref)) + "\n" + str(ref) + "\nRows: " + str(len(ref)) + "\tCols: " + str(len(ref[0])))
		return ref
	
	def loadFromText(self, fileName, delim):
		print("(Loading data...)")
		self.rawData = numpy.genfromtxt(fileName, dtype=None, delimiter=delim)

		# get categories
		csv_reader = csv.reader(open(fileName), delimiter=delim, quotechar='"')
		self.categories = csv_reader.next()
		# remove header (already tried using skip_header in genfromtxt, but for some reason, shape variable doesn't work with that...)
		self.rawData = numpy.delete(self.rawData, 0, 0)

		# Get dimensions
		self.dataRows = self.rawData.shape[0]
		self.dataCols = self.rawData.shape[1]
		
		print("(Data loaded!)")
		print("DATA SIZE: " + str(self.dataRows) + " rows " + str(self.dataCols) + " cols")
		print("Categories:")
		for cat in self.categories:
			print("\t" + cat)

	def normalizeData(self): # normalizes all cols (inputs) and stores it in normalData
	
		print("(Normalizing...)")
		self.normalData = numpy.empty_like(self.rawData)

		for c in range(0, self.dataCols):
			self.normalData[:,c] = self.normalizeVar(self.rawData[:,c])

		print("(Normalizing complete!)")
		
	# normalizes col of data (one input variable)
	# TODO: string normalization is using a very unofficial method and should be worked on further!
	def normalizeVar(self, rawCol):
	
		col = numpy.copy(rawCol) # make a copy of the data to play with
	
		# remove quotes
		for i in range(0, self.dataRows):
			col[i] = col[i].replace('"', '').strip() # TODO: escaped quotes are allowed!
		
		# check if num
		sampleEntry = col[0]
		print("checking col type with sample " + str(sampleEntry)) # DEBUG
		
		if self.isNumber(sampleEntry) == False: # explicitly checking false cause idk how to do basic negation in python?? 
			print("--NORMALIZING STRING COLUMN--") # DEBUG
			
			for i in range(0, self.dataRows):
				# right now, just add up all ascii values of char strings
				stringSum = 0
				for j in range(0, len(col[i])):
					stringSum += ord(col[i][j])

				col[i] = stringSum

		# "cast" so numpy doesn't complain
		col = col.astype(float)

		# numpy magic!!! 
		colSum = col.sum()
		colMean = col.mean()
		sDev = col.std()
		
		# adjust all col values
		for i in range(0, self.dataRows):
			col[i] = (col[i] - colMean) / sDev
			
		return col
		
	def getRawData(self):
		return self.rawData

	def getNormalData(self):
		return self.normalData

	def isNumber(self, num): # http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float-in-python
		try:
			float(num)
			return True
		except ValueError:
			return False
			
print("Starting Engine Test")
inputs = Dataset()
inputs.loadFromText(".\\TestData\\student-mat.csv", ";")
inputs.normalizeData()
type = Datatype()
type.name = "Student Data"
type.rank = inputs.dataCols
inputs.datatype = type
print("Data successfully loaded\nCreating Algorithm")
algorithm = SupervisedClassifier("00-00")
print("Algorithm successfully built")
S = inputs.getColumn("internet", True)
#print("Column " + str(S))
T = inputs.excludeColumn("internet", True)
print("Column " + str(T))
print("Training Algorithm")
metrics = algorithm.fit(T, S)
print("Algorithm trained")
		
#print("nothing broken")
