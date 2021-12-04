import numpy as np
from numpy import linalg
import pandas as pd
from matplotlib import pyplot as plt
import cvxopt
from matplotlib import cm
from mpl_toolkits import mplot3d


def f(x,w,b):
    # wX + b = 0
    return (-b - w[0]*x) / w[1]


def gaussian_kernel(x, y, sigma = 1):
    return np.exp(-linalg.norm(x-y)**2/(2*(sigma**2)))


class Svm(object):

#   https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    def fit(self, X, y, C=0, sigma=1):
        m, n = X.shape

        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = gaussian_kernel(X[i], X[j], sigma)

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
        self.w = None

    def project(self, X, sigma):
        y_predict = np.zeros(len(X))
        for i in range(len(X)): #or m
            for alfa, sv_y, sv_x in zip(self.alfas, self.sv_y, self.sv_x):
                y_predict[i] += alfa * sv_y * gaussian_kernel(X[i], sv_x, sigma)
        return y_predict + self.b

    def predict(self, X, sigma):
        return np.sign(self.project(X, sigma))


# main:
dataset = pd.read_csv("svmData_nls.csv", header=None)
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
X_Test = X_stand[int(length * 5. / 6):, :]
X_Train = X_stand[:int(length * 5. / 6), :]
Y_Test = Y[int(length * 5. / 6):]
Y_Train = Y[:int(length * 5. / 6)]

svm = Svm()
C_target_values = np.linspace(0.1, 10, 10)
Sigma_target_values = np.linspace(0.1, 10, 10)
accuracy = np.zeros((len(C_target_values), len(Sigma_target_values)))
max_accuracy = 0
for i in range(len(C_target_values)):
    for j in range(len(Sigma_target_values)):
        svm.fit(X_Train, Y_Train, C_target_values[i], Sigma_target_values[j])
        y_trained = svm.predict(X_Test, Sigma_target_values[j])
        correct = np.sum(y_trained == Y_Test)
        local_accuracy = correct * 1. / len(y_trained) * 100
        if local_accuracy > max_accuracy:
            max_accuracy = local_accuracy
            const_sigma = C_target_values[i]
            const_c = Sigma_target_values[j]
        accuracy[i][j] = local_accuracy

print("c&sigma&max_acc:", const_c, const_sigma,max_accuracy)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(Sigma_target_values, C_target_values, accuracy, 50, cmap=cm.coolwarm)
ax.set_xlabel('sigma')
ax.set_ylabel('c')
ax.set_zlabel('Accuracy [%]')
plt.show()


svm.fit(X_Train, Y_Train, const_c, const_sigma)
x_graph = np.linspace(-2, 2, 50)
y_graph = svm.project(x_graph, const_sigma)

plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(X_Train[np.where(Y_Train == -1),0], X_Train[np.where(Y_Train == -1),1], color="red")
plt.scatter(X_Train[np.where(Y_Train == 1),0], X_Train[np.where(Y_Train == 1),1], color="blue")
plt.plot(x_graph, y_graph, color="black")
plt.show()
