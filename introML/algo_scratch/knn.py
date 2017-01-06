import csv
import random
import math

# MachineLearningMastery.com Algo Tutorial

with open('iris.data', 'rb') as csvfile:
	lines = csv.reader(csvfile)
		for row in lines:
			print ', '.join(row)

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

