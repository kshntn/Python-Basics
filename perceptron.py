import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

num_dataset = np.array([[0.22, 0.34, 0], [0.21, 0.37, 0], [0.25, 0.31, 0], [0.76, 0.19, 1], [0.84, 0.14, 1]])
features = num_dataset[:, :2]
labels = num_dataset[:, 2].reshape((num_dataset.shape[0], 1))
plt.scatter(features[:, 0], features[:, 1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input Data')
plt.show()
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1
num_output=labels.shape[1]
dim1=[dim1_min,dim1_max]
dim2=[dim2_min,dim2_max]
perceptron=nl.net.newp([dim1,dim2],num_output)
error_progress=perceptron.train(features,labels,epochs=100,show=20,lr=0.03)

plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training Error')
plt.title('Training error progress')
plt.grid()
plt.show()
a=perceptron.sim([[.81,.23]])
print a
b=perceptron.sim([[.27,.42]])
print b