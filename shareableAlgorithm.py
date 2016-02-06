import joblib #THIS COULD BE VERY INSECURE, CHECK WITH SOMEONE WHO ACTUALLY KNOWS WHAT THEY'RE TALKING ABOUT!

class Sharable():
	
	#member variables
	algorithm = None #the trained algorithm object
	name = ""
	version = ""
	statistics = None #this will probably be an sklearn helper class. We might want to wrap it up with other stuff also
	computation = None #we might have to implement this in numpy
	client_history = None #client-side provenance
	usage = None #input and output types, along with usage intentions
	
	def _init_(self, algorithm, name, version, statistics, computation, client_history, usage):
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