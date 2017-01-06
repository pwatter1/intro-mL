import csv
import random
import math
import operator

# MachineLearningMastery.com KNN Algo Tutorial

# load a dataset and split into train & test sets
def loadDataset(filename, split, trainingSet=[], testSet=[]):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)

		for x in range(len(dataset)-1):

			for y in range(4):
				dataset[x][y] = float(dataset[x][y])

			if random.random() < split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])

# calculate similarity 
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)

	return math.sqrt(distance)

# test the euclideanDistance function
data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print 'Distance: ' + repr(distance)

# locate most similar neighbours
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	neighbors = []
	length = len(testInstance)-1

	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))

	distances.sort(key=operator.itemgetter(1))

	for x in range(k):
		neighbors.append(distances[x][0])

	return neighbors

# test the getNeighbors function 
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbors = getNeighbors(trainSet, testInstance, 1)
print(neighbors)

# summarize a prediction from neighbors
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]

		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1

	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)

	return sortedVotes[0][0]

# test the getResponse function
neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
response = getResponse(neighbors)
print(response)

# calculate accuracy of predictions
def getAccuracy(testSet, predictions):
	correct = 0

	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct += 1

	return (correct / float(len(testSet))) * 100.0

# test the getAccuracy function
testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.67
	loadDataset('iris.data', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))

	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()