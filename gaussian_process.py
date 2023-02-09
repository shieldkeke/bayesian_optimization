import numpy as np
import matplotlib.pyplot as plt

class GPR(object):
    def __init__(self):
        self.is_fit = False
        self.trian_x,self.trian_y = None,None
        self.params = {"l":0.5,"sigma_f":0.2}
    
    def fit(self,trian_x,trian_y):
        self.trian_x,self.trian_y = np.array(trian_x),np.array(trian_y)
        self.is_fit=True
    
    def predict(self,test_x):
        # assert self.is_fit,"please fit first"
        test_x = np.array(test_x)
        kfy = self.gaussian_kernel(self.trian_x,test_x)
        kyy = self.gaussian_kernel(test_x,test_x)
        kff = self.gaussian_kernel(self.trian_x,self.trian_x)
        # print(kff)
        kff_inv = np.linalg.inv(kff + 1e-8 * np.eye(len(self.trian_x)))

        mu = kfy.T.dot(kff_inv).dot(self.trian_y)
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

    trian_x = np.arange(-5,5,1).reshape(-1,1)
    trian_y = func(trian_x)

    test_x = np.arange(-5,5,0.1).reshape(-1,1)

    gpr = GPR()
    gpr.fit(trian_x,trian_y)
    mu,cov = gpr.predict(test_x)
    print(cov)
    test_y =  np.ravel(mu)
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    # print(uncertainty)
    plt.figure()
    plt.fill_between(np.ravel(test_x), test_y + uncertainty, test_y - uncertainty, alpha=0.3)
    plt.plot(test_x, test_y, label="predict")
    plt.scatter(trian_x, trian_y, label="train", c="red", marker="x")
    plt.legend()
    plt.show()