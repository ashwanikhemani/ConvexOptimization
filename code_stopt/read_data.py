import numpy

def read_train_sgd():
#function to read train data
	from string import ascii_lowercase
	mapping = list(enumerate(ascii_lowercase))
	mapping = { i[1]:i[0] for i in mapping }

	with open("../data/train_sgd.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	tempX, tempY = [], []
	for row in raw_data[:-1]:
		row = row.split(" ")
		tempY.append( mapping[row[1]])
		tempX.append( numpy.concatenate(\
			( numpy.array(row[5:], dtype=float), numpy.array([1]) )))
		if int(row[2]) < 0:
			dataX.append(numpy.array(tempX))
			dataY.append(numpy.array(tempY, dtype=int))
			tempX, tempY = [], []

	ret = zip(dataX, dataY)
	return list(ret)

def read_test_sgd():
#function to read train data
	from string import ascii_lowercase
	mapping = list(enumerate(ascii_lowercase))
	mapping = { i[1]:i[0] for i in mapping }

	with open("../data/test_sgd.txt", "r") as f:
		raw_data = f.read()
	raw_data = raw_data.split("\n")

	dataX, dataY = [], []
	tempX, tempY = [], []
	for row in raw_data[:-1]:
		row = row.split(" ")
		tempY.append( mapping[row[1]])
		tempX.append( numpy.concatenate(\
			( numpy.array(row[5:], dtype=float), numpy.array([1]) )))
		if int(row[2]) < 0:
			dataX.append(numpy.array(tempX))
			dataY.append(numpy.array(tempY, dtype=int))
			tempX, tempY = [], []

	ret = zip(dataX, dataY)
	return list(ret)
