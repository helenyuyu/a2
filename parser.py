import matplotlib.pyplot as plt
import math
import numpy
import sys
import bisect
from sklearn import linear_model
from sklearn import neighbors
from sklearn.metrics import r2_score
from sklearn import ensemble

def contains(a, target):
	i = bisect.bisect_left(a, target)
	if i != len(a) and a[i] == target:
		return True
	return False
	

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




f_on_island = open('data/intersections.bed', 'r')
on_island = f_on_island.readlines()
on_island = [l.split() for l in on_island]
on_island = numpy.array(on_island)
on_island = on_island[:,1]
on_island = [int(l) for l in on_island]
on_island = numpy.reshape(on_island, (-1))


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

known_vector_rf = []
known_target_rf = []
unknown_vector_rf = []
unknown_target_rf = []

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
	row = train[i][4:37]
	fill_in_row(row)
	if l[5] == 1 and not math.isnan(l[4]) :
		known_target.append(l[4])
		known_vector.append(row)
		known_target_rf.append(l[4])
		if contains(on_island, l[1]):
			    known_vector_rf.append(row + [True])
		else:
			    known_vector_rf.append(row + [False])
		
	if l[5] == 0 and not math.isnan(test[i]):
		unknown_target.append(test[i])
		unknown_vector.append(row)
		unknown_target_rf.append(test[i])
		if contains(on_island, l[1]):
			    unknown_vector_rf.append(row + [True])
		else:
			    unknown_vector_rf.append(row + [False])
	i = i+1	



plot = False

models = [#("Linear regression", linear_model.LinearRegression()), 
	#("Ridge regression", linear_model.Ridge(alpha = .5)),
	#("Lasso regression", linear_model.Lasso(alpha = .1)),
	#("KNN-P", neighbors.KNeighborsRegressor()),
	#("Random Forest", neighbors.KNeighborsRegressor())
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
	print('R^2 of %s: %.8f' % (model[0], r2_score(unknown_target, prediction)))
	if model[0] == "Linear regression" or model[0] == "Ridge regression" or model[0] == "Lasso regression": 
		print model[1].coef_
	
forest = True
if (forest):
	regr = ensemble.RandomForestRegressor()
	regr.fit(known_vector_rf, known_target_rf)
	prediction = regr.predict(unknown_vector_rf)
	rmsd = (numpy.mean((prediction - unknown_target_rf) ** 2)) ** .5
	print("RMSD of %s: %.8f" % ("Random Forest with islands", rmsd))
	print('R^2 of %s: %.8f' % ("Random Forest with islands", r2_score(unknown_target_rf, prediction)))



knn_s = False
if (knn_s):
	x_train = numpy.transpose(known_vector)
	x_test = numpy.transpose(known_target)
	y_train = numpy.transpose(unknown_vector) 
	y_test = numpy.transpose(unknown_target)

	regr = neighbors.KNeighborsRegressor()
	regr.fit(x_train, y_train)
	prediction = regr.predict(x_test)
	rmsd = (numpy.mean((prediction - y_test) ** 2)) ** .5
	print("RMSD of %s: %.8f" % ("KNN-S", rmsd))
	print('R^2 of %s: %.8f' % ("KNN-S", r2_score(numpy.transpose(y_test), numpy.transpose(prediction))))


