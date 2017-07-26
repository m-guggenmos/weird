WeiRD stands for "Weighted Robust Distance" and is a fast and simple classification algorithm
which assigns class labels based on the distance to class prototypes. The distance is the
Manhattan or Euclidian distance between a current sample and a prototype in a space, in which
each feature dimension is scaled by the two-sample t-value of the respective feature in the
training data. Class prototypes correspond to the arithmetic prototypes of each feature in the
training data. The current implementation works for two-class problems only.
__________________________________________________________________________
Matthias Guggenmos, Katharina Schmack and Philipp Sterzer, "WeiRD - a fast and performant
multivariate pattern classifier," 2016 International Workshop on Pattern Recognition in
Neuroimaging (PRNI), Trento, Italy, 2016, pp. 1-4. doi: 10.1109/PRNI.2016.7552349

**Simple example for Python:**

```python
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
```
