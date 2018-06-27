fileA = "test1 0.96116.txt"
fileB = "test3 0.96440.txt"

for i, j in zip(open(fileA, 'r').read().split('\n'), open(fileB, 'r').read().split('\n')):
	if i != j:
		print(i)