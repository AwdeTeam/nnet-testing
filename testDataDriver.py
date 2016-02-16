import SupervisedCategorization as sc

thing = sc.Dataset()

print("(Loading data...)")
thing.loadFromText("TestData/student-mat.csv", ";")
print("(Data loaded!)")

dataMatrixRaw = thing.getRawData()
print(dataMatrixRaw)

dataMatrixNormal = thing.getNormalData()
print(dataMatrixNormal)
