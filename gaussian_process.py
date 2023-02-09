import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class GPR(object):
    def __init__(self):
        self.is_fit = False
        self.train_x,self.train_y = None,None
        self.params = {"l":0.5,"sigma_f":0.2}
    
    def fit(self,train_x,train_y,optimize = False):
        # store train data
        self.train_x,self.train_y = np.array(train_x),np.array(train_y)
        self.is_fit=True

        # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.gaussian_kernel(self.train_x, self.train_x) + 1e-8 * np.eye(len(self.train_x))
            loss = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_x) * np.log(2 * np.pi)
            return loss.ravel()
        
        if optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                    bounds=((1e-4, 1e4), (1e-4, 1e4)),
                    method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

    def predict(self,test_x):
        # assert self.is_fit,"please fit first"
        test_x = np.array(test_x)
        kfy = self.gaussian_kernel(self.train_x,test_x)
        kyy = self.gaussian_kernel(test_x,test_x)
        kff = self.gaussian_kernel(self.train_x,self.train_x)
        # print(kff)
        kff_inv = np.linalg.inv(kff + 1e-8 * np.eye(len(self.train_x)))

        mu = kfy.T.dot(kff_inv).dot(self.train_y)
        cov = kyy-kfy.T.dot(kff_inv).dot(kfy)
        return mu,cov


    def gaussian_kernel(self,x1,x2):
        # print(x1)
        # print(x2)
        d = np.sum(x1**2,1).reshape(-1,1) + np.sum(x2**2,1) - 2*np.dot(x1,x2.T)
        return self.params['sigma_f']**2 * np.exp(-0.5/self.params['l']**2*d)

if __name__ == '__main__':
    def func(x):
        x = np.asarray(x)
        y = np.sin(x)+np.random.normal(0,0.01,size=x.shape)
        return y

    train_x = np.arange(-5,5,1).reshape(-1,1)
    train_y = func(train_x)

    test_x = np.arange(-5,5,0.1).reshape(-1,1)

    gpr = GPR()
    gpr.fit(train_x,train_y)
    mu,cov = gpr.predict(test_x)
    print(cov)
    test_y =  np.ravel(mu)
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    # print(uncertainty)
    plt.figure()
    plt.fill_between(np.ravel(test_x), test_y + uncertainty, test_y - uncertainty, alpha=0.3)
    plt.plot(test_x, test_y, label="predict")
    plt.scatter(train_x, train_y, label="train", c="red", marker="x")
    plt.legend()
    plt.show()