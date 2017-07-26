WeiRD stands for "Weighted Robust Distance" and is a fast and simple classification algorithm
that assigns class labels based on the distance to class prototypes. The distance is the
Manhattan or Euclidian distance between a current sample and a prototype in a space, in which
each feature dimension is scaled by the two-sample t-value of the respective feature in the
training data. Class prototypes correspond to the arithmetic prototypes of each feature in the
training data. The current implementation works for two-class problems only.
__________________________________________________________________________
Matthias Guggenmos, Katharina Schmack and Philipp Sterzer, "WeiRD - a fast and performant
multivariate pattern classifier," 2016 International Workshop on Pattern Recognition in
Neuroimaging (PRNI), Trento, Italy, 2016, pp. 1-4. doi: 10.1109/PRNI.2016.7552349
