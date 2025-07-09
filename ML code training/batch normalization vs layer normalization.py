import numpy as np
#------------------------
# in this doc, I am going to build batch normalization and layer normalization from scratch; I will also 
# compare batch normalization with and without normalization and layer normalization with and without 
# normalization
#------------------------

np.random.seed(0)
X = np.random.randn(10,5)
print(X)
true_w = np.random.randn(5,1)
y = X @ true_w + np.random.randn(10,1)*0.1


class BatchNorm1d:
    def __init__(self, num_features, eps = 1e-5):
        self.eps = eps
        self.gamma = np.ones((1,num_features))
        self.beta = np.zeros((1,num_features))
    
    def forward(self,x):
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.var = np.var(x, axis=0, keepdims=True)
        self.x_hat = (x-self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

class LayerNorm:
    def __init__(self, num_features, eps = 1e-5):
        self.eps = eps
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))

    def forward(self,x):
        self.mean = np.mean(x, axis=1,keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta
        
def predict(x, weights):
    return x @ weights

def mse_loss(pred, target):
    return np.mean((pred - target)**2)

def l2_regularization(weights, lambda_):
    return lambda_ * np.sum(weights ** 2)

w = np.random.randn(5,1)

bn = BatchNorm1d(X.shape[1])
ln = LayerNorm(X.shape[1])
print(bn)
print(ln)

X_bn = bn.forward(X)
X_ln = ln.forward(X)
print(X_bn)
print(X_ln)


# 1. BatchNorm wo regularization
predict_bn = predict(X_bn, w)
loss_bn = mse_loss(predict_bn, y)

# 2. Batchnorm with L2 regularization
loss_bn_reg = mse_loss(predict_bn,y) + l2_regularization(w,lambda_=0.1)

# 3. LayerNorm without regularization
predict_ln = predict(X_ln, w)
loss_ln = mse_loss(predict_ln, y)

# 4. layerNorm with L2 regularization
loss_ln_reg = mse_loss(predict_ln, y) + l2_regularization(w, lambda_= 0.1)



print(f"BN w/o reg loss: {loss_bn:.4f}")
print(f"BN w/  reg loss: {loss_bn_reg:.4f}")
print(f"LN w/o reg loss: {loss_ln:.4f}")
print(f"LN w/  reg loss: {loss_ln_reg:.4f}")