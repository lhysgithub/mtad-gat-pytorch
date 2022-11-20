import numpy as np

x_train = np.random.random((5,5))
feature_numbers = 3
sellected = range(feature_numbers)
temp_x_train = np.array([])
for i in range(len(sellected)):
    xi = x_train[:][sellected[i]].reshape(-1, 1)
    if i == 0:
        temp_x_train = xi
    else:
        temp_x_train = np.concatenate((temp_x_train,xi),axis = 1)
print(temp_x_train)