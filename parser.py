import matplotlib.pyplot as plt
import math
import numpy
import sys
from sklearn import linear_model
from sklearn import neighbors
from sklearn.metrics import r2_score

def find_nearest(sample):
	nearest = []
	last_value = -1
	last_position = sys.maxint
	for l in sample:
		current_position = l[1]
		if (not math.isnan(l[4])):
			last_value = l[4]
			last_position = l[2]
		nearest.append((last_value, abs(last_position-current_position)))
		last_value = -1
	last_position = -sys.maxint-1
	for i in reversed(xrange(len(sample))):
		l = sample[i]
		current_position = l[1]
		if (not math.isnan(l[4])):
			last_value = l[4]
			last_position = l[2]
		if (abs(last_position - current_position) < nearest[i][1]):
			nearest[i] = ((last_value, abs(last_position-current_position)))
	return nearest;


def fill_in_row(row):
	count = 0
	sum = 0
	for i in row:
		if not math.isnan(i):
			count+=1
			sum+=i
	avg = sum/count
	for i in range(0, len(row)):
		if math.isnan(row[i]):
			row[i] = avg
	return row


filename_train = 'data/intersected_final_chr1_cutoff_20_train_revised.bed'

f_train = open(filename_train,'r')

train = f_train.readlines()

train = [l.split() for l in train]

for l in train:
	for i in range(1,3):
		l[i] = int(l[i])
	if (l[3] == '+'):
		l[3] = True
	else:
		l[3] = False
	for i in range(4, len(l)-1):
		l[i] = float(l[i])
	l[len(l)-1] = int(l[len(l)-1])


	
filename_test = 'data/intersected_final_chr1_cutoff_20_test.bed'
f_test = open(filename_test,'r')
test = f_test.readlines()
test = [float(l.split()[4]) for l in test]



filename_sample = 'data/intersected_final_chr1_cutoff_20_sample.bed'
f_sample = open(filename_sample,'r')

sample = f_sample.readlines()
sample = [l.split() for l in sample]


known_vector = []
known_target = []
unknown_vector = []
unknown_target = []
i = 0


for l in sample:
	for j in range(1,3):
		l[j] = int(l[j])
	if (l[3] == '+'):
		l[3] = True
	else:
		l[3] = False
	l[4] = float(l[4])
	l[5] = int(l[5])
	row = train[i][4:35]
	fill_in_row(row)
	if l[5] == 1 and not math.isnan(l[4]) :
		known_target.append(l[4])
		known_vector.append(row)
	if l[5] == 0 and not math.isnan(test[i]):
		unknown_target.append(test[i])
		unknown_vector.append(row)
	i = i+1	


plot = False
KNN = True
if (KNN):
	x_train = numpy.transpose(known_vector)
	x_test = numpy.transpose(known_target)
	y_train = numpy.transpose(unknown_vector) 
	y_test = numpy.transpose(unknown_target)

	regr = neighbors.KNeighborsRegressor()
	regr.fit(x_train, y_train)
	prediction = regr.predict(x_test)
	rmsd = (numpy.mean((prediction - y_test) ** 2)) ** .5
	print("RMSD of %s: %.8f" % ("KNN", rmsd))
	print('R^2 of %s: %.8f' % ("KNN", r2_score(numpy.transpose(prediction), numpy.transpose(y_test))))


models = [#("Linear regression", linear_model.LinearRegression()), 
	#("Ridge regression", linear_model.Ridge(alpha = .5))
	#("Lasso regression", linear_model.Lasso(alpha = 1))
	]

for model in models:
	regr = model[1]
	regr.fit(known_vector, known_target)
	prediction = regr.predict(unknown_vector)
	 
 # root mean square deviation 
	rmsd = (numpy.mean((prediction - unknown_target) ** 2)) ** .5
	   
	if (plot): 
		error = numpy.array(prediction) - numpy.array(unknown_target)
		plt.hist(error, bins = 100)
		plt.show()
	   
	print("RMSD of %s: %.8f" % (model[0], rmsd))
	print('R^2 of %s: %.8f' % (model[0], r2_score(prediction, unknown_target)))
	if (model[0] != "KNN"):
		print("coefficients")
		print(model[1].coef_)
