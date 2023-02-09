from gaussian_process import GPR
import numpy as np
from scipy.stats import norm 

class bayesianOpt:
    def __init__(self) -> None:
        self.surrogate = GPR()
        self.bounds = np.array([])
        self.ymax = -100
        self.x = []
        self.y = []

    # 3 types of acquisition function
    def PI(self, x, gp, y_max, xi = 0.01): # xi:threshold PI(x)=P(f(x)>=f(x+) + xi)
        mean, cov = gp.predict(x)
        std = np.sqrt(np.diag(cov))
        z = (mean - y_max - xi)/std
        return norm.cdf(z)

    def EI(self, x, gp, y_max, xi = 0.01):
        # derivation: http://ash-aldujaili.github.io/blog/2018/02/01/ei/
        mean, cov = gp.predict(x)
        std = np.sqrt(np.diag(cov))
        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)
    
    def UCB(self, x, gp, k = 1):
        mean, cov = gp.predict(x)
        std = np.sqrt(np.diag(cov))
        return mean + k * std

    def acq_max(self, random_state=666, num=1000):
        # randomly sample
        xs= np.random.RandomState(random_state).uniform(self.bounds[:, 0], self.bounds[:, 1], size=(num, self.bounds.shape[0]))
        # ys = self.EI(xs, self.surrogate, self.ymax)
        ys = self.UCB(xs, self.surrogate)
        x_max = xs[ys.argmax()]
        max_acq = ys.max()
        return x_max #suggestion
    
    def set_bound(self, bounds):
        self.bounds = bounds

    def update(self, x, y):
        # for manually adding one data, next x may not be suggestion(in real world)
        self.x.append(x)
        self.y.append(y)
        self.surrogate.fit(self.x,self.y) #update surrogate function
        suggestion = self.acq_max()
        return suggestion

    def result(self):
        return self.x[np.array(self.y).argmax()], np.array(self.y).max()

    def optimize(self, fun, n_iters, x_init = [], y_init = []):
        # for auto optimize, and next x = suggestion
        self.x.extend(x_init)
        self.y.extend(y_init)
        print(self.x)
        for i in range(n_iters):
            if len(self.x)>0:
                self.surrogate.fit(self.x,self.y)
                suggestion = self.acq_max()
            else:
                suggestion = np.random.RandomState(666).uniform(self.bounds[:, 0], self.bounds[:, 1], size=(1, self.bounds.shape[0]))
            self.x.append(suggestion.tolist())
            self.y.append(fun(suggestion))
            self.ymax = max(self.y)
        return self.result()

if __name__ == "__main__":
    import math
    fun = math.sin
    b = bayesianOpt()
    bound = np.array([[0,2]])
    b.set_bound(bound)
    # first usage
    x, y = b.optimize(fun, 20, [[0], [0.5]],[math.sin(0), math.sin(0.5)])
    print(x, y)

    # second usage
    b = bayesianOpt()
    bound = np.array([[0,2]])
    b.set_bound(bound)
    b.update([0],0)
    suggestion = b.update([0.5], math.sin(0.5))
    for i in range(20):
        # you can update it even if you did't put the suggestion into the function(in the real world)
        suggestion = b.update(suggestion.tolist(), fun(suggestion)) 
    print(b.result())