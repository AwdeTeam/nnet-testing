import SupervisedCategorization as sc

data = sc.Dataset()

data.loadFromText("TestData/student-mat.csv", ";")
data.normalizeData()

print("\n\n*******************************************************************")
print("\t\tRAW DATA")
print("*******************************************************************")
dataMatrixRaw = data.getRawData()
print(dataMatrixRaw)
print("-------------------------------------------------------------------")

print("\n\n*******************************************************************")
print("\t\tNORMALIZED")
print("*******************************************************************")

dataMatrixNormal = data.getNormalData()
print(dataMatrixNormal)
print("-------------------------------------------------------------------")
