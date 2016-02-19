print("(Running imports...)")
import math
import numpy
import sklearn
#import metrics from sklearn
print("(Finished imports.)")

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

	dataRows = 0
	dataCols = 0
	
	categories = [] #keep track of the name of the categories
	type = "real"
	bound = [-1, 1]
	
	def loadFromText(self, file, delim):
		#self.rawData = numpy.loadtxt(open(file,"rb"), delimiter=delim, skiprows=0, ndmin=2)
		#self.rawData = numpy.loadtxt(open(file,"rb"), dtype=<type 'string'>, delimiter=delim, skiprows=0, ndmin=2)
		print("(Loading data...)")
		self.rawData = numpy.genfromtxt(file, dtype=None, delimiter=delim)

		# remove header (already tried using skip_header in genfromtxt, but for some reason, shape variable doesn't work with that...)
		self.rawData = numpy.delete(self.rawData, 0, 0)
		

		# Get dimensions
		self.dataRows = self.rawData.shape[0]
		self.dataCols = self.rawData.shape[1]
		
		print("(Data loaded!)")
		print ("DATA SIZE: " + str(self.dataRows) + " rows " + str(self.dataCols) + " cols")

		
		#for t in range (0, colsInRawData)
			#categories[t] = categories[0][t]
			
		#for i in range (1, rowsInRawData):
			#for j in range (0, colsInRawData):
				#normalData[i][j] = rawData[i][j]

	# normalization types: simple
	def normalizeData(self): # pass in a column of data and it will return normalized (should also auto assign to the normaldata thing) NOTE: don't actually return, just edit normalData!
	
		print("(Normalizing...)")
		self.normalData = numpy.empty_like(self.rawData)

		#for r in range(0, self.dataRows):
		#	for c in range(0, self.dataCols):
		#		self.normalData[r,c] = self.normalizeEntry(self.rawData[r,c])

		for c in range(0, self.dataCols):
			self.normalData[:,c] = self.normalizeVar(self.rawData[:,c])
					
		print("(Normalizing complete!)")
		
	# normalizes row of data (one input variable)
	def normalizeVar(self, row):
	
		# remove quotes
		for i in range(0, self.dataRows):
			row[i] = row[i].replace('"', '').strip()
		
		# check if num
		sampleEntry = row[0]
		print("checking col type with sample " + str(sampleEntry))
		
		if self.isNumber(sampleEntry) == False: # explicitly checking false cause idk how to do basic negation in python?? 
			print("Normalizing string column")
			return row # at some point, don't return, just turn strings into numbers

		# "cast" so numpy doesn't complain
		row = row.astype(float)

		# get mean
		rowSum = row.sum()
		rowMean = row.mean()
		sDev = row.std()
		#rowMean = rowSum / self.dataRows
		
		# find standard deviation
		#sDev = 0
		#summedTerm = 0
		#for i in range(0, self.dataRows):
		#	summedTerm += (row[i] - rowMean)**2

		#sDev = sqrt(summedTerm / self.dataRows)

		# adjust all row values
		for i in range(0, self.dataRows):
			row[i] = (row[i] - rowMean) / sDev
			
		return row

	def normalizeEntry(self, entry): # normalizes single thing (checks for data type, numberizes accordingly)
		# print("[normalizing " + str(entry) + "]") # DEBUG
		
		# remove quotes
		entry = entry.replace('"', '').strip()
		# print("[[now " + str(entry) + "]]") # DEBUG

		# if number, make invert
		if self.isNumber(entry):
			# print("[[is number]]") # DEBUG
			entry = float(entry)
			if entry == 0: # don't divide by zero!!! Python doesn't like it...
				return 0
			return (1 / entry)

		# if string, do some magic
		
		return 1.0
		
		
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
		
#print("nothing broken")
