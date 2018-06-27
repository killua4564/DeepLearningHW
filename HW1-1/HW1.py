import numpy

a = open("answer.txt", "r").read()
b = open("data.txt", "r").read()

a = numpy.array([list(map(int, a.split(' ')))])
b = numpy.array([list(map(int, i.split(' '))) for i in b.strip('\n').split('\n')])

a = numpy.transpose(a)

print(a + b)