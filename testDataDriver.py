import SupervisedCategorization as sc

data = sc.Dataset()

data.loadFromText("TestData/student-mat.csv", ";")
data.normalizeData()

dataMatrixRaw = data.getRawData()
print(dataMatrixRaw)

dataMatrixNormal = data.getNormalData()
print(dataMatrixNormal)
