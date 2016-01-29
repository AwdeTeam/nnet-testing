import NNet as net

nn = net.NeuralNetwork(2,1,1)
#nn.initTheanoFunctions()
nn.readTrainingData("data.txt")
