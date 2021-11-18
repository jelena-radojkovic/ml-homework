import numpy as np
import pandas as pd
import math


def calculate_probability(x, mean, stdev):
    # calculate the class probability using gaussian distribution
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def predict(num_classes, x, mi, std, fi):
    y_pred = np.zeros(len(x))
    for i in range(len(x)):
        pr = np.ones(num_classes)
        for j in range(x.shape[1]):
            pr[0] = pr[0] * calculate_probability(x[i, j], mi[0, j], std[0, j])
            pr[1] = pr[1] * calculate_probability(x[i, j], mi[1, j], std[1, j])
            pr[2] = pr[2] * calculate_probability(x[i, j], mi[2, j], std[2, j])
        for j in range(num_classes):
            pr[j] = pr[j] * fi[j]
        y_pred[i] = np.argmax(pr)
    return y_pred


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

# data for testing and training
# first 5/6 are for training, last 1/6 are for testing purposes
num_classes = len(np.unique(Y))
length = len(X)
X_Test = X_stand[int(length * 5. / 6):, :]
X_Train = X_stand[:int(length * 5. / 6), :]
Y_Test = Y[int(length * 5. / 6):]
Y_Train = Y[:int(length * 5. / 6)]

x_train_0 = (X_Train[np.where(Y_Train == 0), :])
x_train_1 = (X_Train[np.where(Y_Train == 1), :])
x_train_2 = (X_Train[np.where(Y_Train == 2), :])

# calculating means
means = np.zeros((3, X_Train.shape[1]))
means[0] = np.mean(x_train_0, axis=0)[0]
means[1] = np.mean(x_train_1, axis=0)[0]
means[2] = np.mean(x_train_2, axis=0)[0]

# calculating stds
stds = np.zeros((3, X_Train.shape[1]))
stds[0] = np.std(x_train_0.reshape(-1,5), axis=0)
stds[1] = np.std(x_train_1.reshape(-1,5), axis=0)
stds[2] = np.std(x_train_2.reshape(-1,5), axis=0)

# calculating fis
fis = np.zeros(3)
fis[0] = (1./len(Y_Train))*(len(np.where(Y_Train == 0)[0]))
fis[1] = (1./len(Y_Train))*(len(np.where(Y_Train == 1)[0]))
fis[2] = (1./len(Y_Train))*(len(np.where(Y_Train == 2)[0]))

y_predicted = predict(num_classes, X_Train, means, stds, fis)
correct = np.array(np.where(Y_Train == y_predicted)).ravel()
num_correct = len(correct)
print('Correct: ', num_correct)
print('Wrong: ', len(Y_Train) - num_correct)
print('Accuracy: ', float(num_correct) / len(Y_Train) * 100, '%')

# Test set
y_predicted = predict(num_classes, X_Test, means, stds, fis)
correct = np.array(np.where(Y_Test == y_predicted)).ravel()
num_correct = len(correct)
print('Correct: ', num_correct)
print('Wrong: ', len(Y_Test) - num_correct)
print('Accuracy: ', (float(num_correct)/len(Y_Test)) * 100, '%')