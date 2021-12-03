import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cvxopt


def f(x,w,b):
    # wX + b = 0
    return (-b - w[0]*x) / w[1]


class Svm(object):

#   https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    def fit(self, X, y, C = 0):
        m, n = X.shape

        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = np.dot(X[i], X[j]) # linear kernel

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(m) * (-1))
        A = cvxopt.matrix(y, (1, m))
        b = cvxopt.matrix(0.0)

        G = cvxopt.matrix(np.vstack((np.diag(np.ones(m) * -1), np.identity(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * C)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # alfa
        alfas = np.ravel(solution['x'])

        # support vectors
        sv = alfas > 1e-5
        ind = np.arange(len(alfas))[sv]
        self.alfas = alfas[sv]
        self.sv_x = X[sv]
        self.sv_y = y[sv]

        # b
        self.b = 0
        for i in range(len(self.alfas)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.alfas * self.sv_y * K[ind[i], sv])
        self.b /= len(self.alfas)

        # w
        self.w = np.zeros(n)
        for i in range(len(self.alfas)):
            self.w += self.alfas[i] * self.sv_y[i] * self.sv_x[i]


    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.project(X))


# main:
dataset = pd.read_csv("svmData_ls.csv", header=None)
dataset = dataset.sample(frac=1).reset_index(drop=True)

# predictors:
X = dataset.values[:, :2]

# output:
Y = dataset.values[:, 2]

# standardization:
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_stand = (X - X_mean) / X_std

# data for testing and training
# first 5/6 are for training, last 1/6 are for testing purposes
length = len(X)
X_Test = X_stand[int(length  / 2):, :]
X_Train = X_stand[:int(length / 2), :]
Y_Test = Y[int(length / 2):]
Y_Train = Y[:int(length / 2)]

svm = Svm()
C_target_values = np.linspace(0.1, 50, 10)
accuracy = []
for c in C_target_values:
    svm.fit(X_Train, Y_Train, c)
    y_trained = svm.predict(X_Test)
    correct = np.sum(y_trained == Y_Test)
    accuracy.append(correct * 1. / len(y_trained) * 100)

plt.xlabel("c")
plt.ylabel("accuracy")
plt.plot(C_target_values, accuracy)
plt.show()

c = C_target_values[np.where(accuracy == max(accuracy))]
print("best C is:", c[0])
svm.fit(X_Train, Y_Train, c[0])
y_trained = svm.predict(X_Test)

x_graph = np.linspace(-2, 2, 2)
y_graph = f(x_graph, svm.w, svm.b)

plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(X_Train[np.where(Y_Train == -1),0], X_Train[np.where(Y_Train == -1),1], color="red")
plt.scatter(X_Train[np.where(Y_Train == 1),0], X_Train[np.where(Y_Train == 1),1], color="blue")
plt.plot(x_graph, y_graph, color="black")
plt.show()