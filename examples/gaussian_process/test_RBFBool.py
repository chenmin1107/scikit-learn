import numpy as np
from matplotlib import pyplot as plt
from math import *

from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBFBool
from sklearn.datasets import fetch_mldata
import sys

# generate some test data to test the new kernel defined for Intersection
# navigation
X = []
y = []
for i in np.linspace(0,10,10):
    X.append([i, 0, 0, 0, 0, 0, 0])
    # y.append(i * 2)
    y.append(0.1 * sin(i))

# preproscessing
scaler = preprocessing.StandardScaler().fit(X)
print scaler
XT = scaler.transform(X)
XT = X
print 'scaler mean: ', scaler.mean_
print 'scaler scale_: ', scaler.scale_

# display data
a = input('Do you want to display test data ? ')
if a == 'y':
    print 'X: ', XT
    print 'y: ', y

# initialize the kernel 
kernel_test = RBFBool(num_pattern = 1, length_scale = [2, 2, 2, 2], length_scale_bounds = [1, 10])

print 'after kernel init'

# train the model
gp_test = GaussianProcessRegressor(kernel=kernel_test, alpha=0.0, normalize_y=True)

print 'after GP regressor'
# print("GPML kernel: %s" % gp_test.kernel_)
# print("Log-marginal-likelihood: %.3f"
#       % gp_test.log_marginal_likelihood(gp_test.kernel_.theta))

gp_test.fit(XT, y)

print("GPML kernel: %s" % gp_test.kernel_)
print("Log-marginal-likelihood: %.3f"
      % gp_test.log_marginal_likelihood(gp_test.kernel_.theta))


X_ = []
for i in range(15):
    X_.append([i+0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# XT_ = scaler.transform(X_)
XT_ = X_
print 'X_ ', XT_
y_pred, y_std = gp_test.predict(XT_, return_std=True)
print 'y_pred: ', y_pred

# Plot the predict result
X = np.array(X)
y = np.array(y)
X_ = np.array(X_)
plt.scatter(X[:, 0], y, c='k')
plt.plot(X_[:, 0], y_pred)
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std, alpha = 0.5, color='k')

plt.xlim(X_[:, 0].min(), X_[:, 0].max())
plt.xlabel("x")
plt.ylabel(r"u")
plt.title(r"Test SquareExpWithBool Kernel")
plt.tight_layout()
plt.show()
