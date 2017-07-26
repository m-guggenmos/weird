n_samples = 500;
n_features = 100;
class_sep = 0.1;
rng(8, 'twister')

X = rand(n_samples, n_features);
addX = [class_sep * repmat(rand(1, n_features), n_samples/2, 1);
        zeros(n_samples/2, n_features)];
X = X + addX;
y = [-ones(n_samples/2, 1); ones(n_samples/2, 1)];

train_ind = repmat([true(n_samples/4, 1); false(n_samples/4, 1)], 2, 1);
test_ind = ~train_ind;

model = weirdtrain(y(train_ind), X(train_ind, :));
predictions = weirdpredict(y(test_ind), X(test_ind, :), model);