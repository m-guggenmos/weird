import numpy as np
from weird import WeiRD

np.random.seed(2)

# parameters
n_samples_per_class = 100
n_features = 20

# create data
X1 = np.random.rand(n_features) + np.random.rand(n_samples_per_class, n_features)
X2 = np.random.rand(n_features) + np.random.rand(n_samples_per_class, n_features)
X_fit = np.vstack((X1, X2))
X_predict = X_fit + np.random.rand(2*n_samples_per_class, n_features)
y = np.hstack((np.zeros(n_samples_per_class), np.ones(n_samples_per_class)))

# perform classification
weird = WeiRD()
weird.fit(X_fit, y)
predictions = weird.predict(X_predict)
print('Classification accuracy = %.1f%%' % (100*np.mean((predictions == y))))
# Should give 93.5% with seed 2
