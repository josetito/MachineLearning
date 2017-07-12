
import idx2numpy


trainImages = idx2numpy.convert_from_file('train-images.idx3-ubyte')
trainLabels = idx2numpy.convert_from_file('train-images.idx3-ubyte')
testImages = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
testLabels = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
print trainImages
print "sapo"
print trainLabels


print "test"
print testImages