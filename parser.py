import matplotlib.pyplot as plt
import math
import numpy
from sklearn import linear_model

def fill_in_row(row):
	sum = 0
	count = 0
	for r in row:
		if not math.isnan(r):
			sum += r
			count += 1
	avg= sum/count
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

prev_row = fill_in_row(train[0][4:35]);
next_row = fill_in_row(train[1][4:35]);
row = prev_row;

# prev_location = train[0][1];
# prev_prev_row;
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
	# row = fill_in_row(train[i][4:35]);
	if (i < len(sample)-1):
		next_row = fill_in_row(train[i+1][4:35])
	else :
		next_row = row

	if l[5] == 1 and not math.isnan(l[4]) :
		known_target.append(l[4])
		known_vector.append(prev_row + row + next_row)
		# known_vector.append(prev_row + row + [l[1] - prev_location])

	if l[5] == 0 and not math.isnan(test[i]):
		unknown_vector.append(prev_row + row + next_row)
		# unknown_vector.append(prev_row + row + [l[1] - prev_location])
		unknown_target.append(test[i])
	prev_row = row
	row = next_row
	prev_location = l[2]
	i = i+1	

print 'starting regression'
# print 'unknown_target: -----------'
# for i in unknown_target:
# 	print i

# print 'unknown_vector: -----------'
# for i in unknown_vector:
# 	print i


# print 'known_target: -----------'
# for i in known_target:
# 	print i

# print 'known_vector: -----------'
# for i in known_vector:
# 	print i


regr = linear_model.LinearRegression()
regr.fit(known_vector, known_target)
prediction = regr.predict(unknown_vector)

# root mean square deviation 
rmsd = (numpy.mean((prediction - unknown_target) ** 2)) ** .5

error = prediction - unknown_target
plt.hist(error, bins = 100)
plt.show()


print("RMSD: %.8f" % rmsd)
print('R^2: %.8f' % regr.score(unknown_vector, unknown_target))
