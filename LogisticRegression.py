import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def create_y_dataset(y, model):
    y_train = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] == model:
            y_train[i] = 1
    return y_train


def calculate_hypothesis(x, theta):
    # h = 1 / (1 + e^(-theta.T * x))
    e = np.exp(-np.matmul(x, theta.T))
    h = 1 / (1 + e)
    return h


def calculate_plausibility(y, h):
    # sum of [y*ln(h) + (1-y)ln(1-h)]
    return np.sum(np.matmul(y, np.log2(h)) + np.matmul(np.subtract(np.ones(len(y)), y), np.log2(np.subtract(np.ones(len(h)), h))), axis=0)


def gradient_descent(x, y, theta, learn_rate, mini_batches):
    len_array = int(len(x) / mini_batches)
    loss = np.zeros(len_array)
    iters = np.zeros(len_array)
    iters[0] = mini_batches
    for i in range(len_array - 1):
        x_temp = x[i * mini_batches:(i + 1) * mini_batches, :]
        y_temp = y[i * mini_batches:(i + 1) * mini_batches]
        h = calculate_hypothesis(x_temp, theta)
        l = calculate_plausibility(y_temp, h)
        loss[i] = -l
        iters[i + 1] = iters[i] + mini_batches
        # theta = theta + learn_rate * gradient
        theta = theta + learn_rate * (np.matmul(x_temp.T, (y_temp - h)))
        # theta = np.sum(theta, learn_rate * (np.matmul(x_temp.T, (y_temp-h))))
    return theta, iters, loss


# main:
dataset = pd.read_csv("multiclass_data.csv", header=None)
dataset = dataset.sample(frac=1).reset_index(drop=True)

# predictors:
X = dataset.values[:, :5]

# output:
Y = dataset.values[:, 5]

# standardization:
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_stand = (X - X_mean) / X_std
X_stand = np.pad(X_stand, ((0, 0), (1, 0)), mode='constant', constant_values=1)

# data for testing and training
# first 5/6 are for training, last 1/6 are for testing purposes
length = len(X)
X_Test = X_stand[int(length * 5. / 6):, :]
X_Train = X_stand[:int(length * 5. / 6), :]
Y_Test = Y[int(length * 5. / 6):]
Y_Train = Y[:int(length * 5. / 6)]

# model 0
theta = np.zeros(np.shape(X_Train)[1])
y_train_0 = create_y_dataset(Y_Train, 0)
# 1x optimals values for learning rate and mini batches
_, iters1, loss1 = gradient_descent(X_Train, y_train_0, theta, 0.5, 10)
# 2x optimal value for learning rate and suboptimal values for mini batches
theta_0, iters2, loss2 = gradient_descent(X_Train, y_train_0, theta, 0.5, 1)
_, iters3, loss3 = gradient_descent(X_Train, y_train_0, theta, 0.5, 40)
# 2x suboptimal values for learning rate and optimal mini batches
_, iters4, loss4 = gradient_descent(X_Train, y_train_0, theta, 0.01, 10)
_, iters5, loss5 = gradient_descent(X_Train, y_train_0, theta, 1, 10)

plt.xlabel("Iteracija")
plt.ylabel("Gubitak")
plt.plot(iters1, loss1, color="red")
plt.plot(iters2, loss2, color="blue")
plt.plot(iters3, loss3, color="green")
plt.plot(iters4, loss4, color="yellow")
plt.plot(iters5, loss5, color="black")
plt.show()

# model 1
y_train_1 = create_y_dataset(Y_Train, 1)
# 1x optimals values for learning rate and mini batches
theta_1, _, _ = gradient_descent(X_Train, y_train_1, theta, 0.5, 2)

# model 2
y_train_2 = create_y_dataset(Y_Train, 2)
# 1x optimals values for learning rate and mini batches
theta_2, _, _ = gradient_descent(X_Train, y_train_2, theta, 0.5, 2)

# testing
y_predicted = np.zeros((len(Y_Test),3))
y_predicted[:,0] = calculate_hypothesis(X_Test, theta_0)
y_predicted[:,1] = calculate_hypothesis(X_Test, theta_1)
y_predicted[:,2] = calculate_hypothesis(X_Test, theta_2)
y_predicted = np.argmax(y_predicted, axis=1)
correct = np.array(np.where(Y_Test == y_predicted)).ravel()
print('Correct: ', len(correct))
print('Wrong: ', len(Y_Test) - len(correct))
print('Accuracy: ', (len(correct)/len(Y_Test)) * 100, '%')

