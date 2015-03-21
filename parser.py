filename = 'data/intersected_final_chr1_cutoff_20_train_revised.bed'

f = open(filename,'r')

train = f.readlines()

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

print train[0]