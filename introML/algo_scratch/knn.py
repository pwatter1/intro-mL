import csv
import random
import math
import operator

# MachineLearningMastery.com Algo Tutorial
'''
with open('iris.data', 'rb') as csvfile:
	lines = csv.reader(csvfile)
	for row in lines:
		print ', '.join(row)
'''
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
		neighbours.append(distances[x][0])

	return neighbours

# test the getNeighbors function 
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbors = getNeighbors(trainSet, testInstance, 1)
print(neighbors)

# summarize a prediction from neighbours
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