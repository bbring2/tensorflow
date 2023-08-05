import numpy

array1d = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
array2d1 = numpy.array([[1, 2, 3, 5], [6, 7, 8, 10], [11, 13, 14, 15], [17, 18, 19, 20]])
array2d2 = numpy.array([[1, 2, 3, 4], [6, 7, 8, 9], [12, 13, 14, 15], [16, 18, 19, 20]])

multiply = numpy.dot(array2d1, array2d2)
transition = numpy.transpose(array2d1)

print(multiply)
print(transition)