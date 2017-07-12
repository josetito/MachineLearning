

#mndata = MNIST('t10k-images.idx3-ubyte')
#images, labels = mndata.load_training()
print "estoy feliz"


#train-images.idx3-ubyte
#t10k-images.idx3-ubyte

from mnist import MNIST
import struct
with open('train-images.idx3-ubyte', 'rb') as f:
    data = f.read(8)

#print(data)

print(struct.unpack('>II', data))


