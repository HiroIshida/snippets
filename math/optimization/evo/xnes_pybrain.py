import numpy as np
from scipy.linalg import expm
from scipy import dot, array, randn, eye, outer, exp, trace, floor, log, sqrt
from numpy import zeros, argmax, array, power, exp, sqrt, var, zeros_like, arange, mean, log

class HansenRanking(object):
    """ Ranking, as used in CMA-ES """
    def __call__(self, R):
        def rankedFitness(R):
            """ produce a linear ranking of the fitnesses in R.
            (The highest rank is the best fitness)"""

            res = zeros_like(R)
            l = list(zip(R, list(range(len(R)))))
            l.sort()
            for i, (_, j) in enumerate(l):
                res[j] = i
            return res

        ranks = rankedFitness(R)
        return array([max(0., x) for x in log(len(R)/2.+1.0)-log(len(R)-array(ranks))])

class XNES(object):
    uniformBaseline = True

    def __init__(self, x_init):
        # set numParameters
        dim = len(x_init)
        self.numParameters = dim
        self.centerLearningRate = 1.0
        self.covLearningRate = 0.6*(3+log(dim))/dim/sqrt(dim)

        pybrain_original = True
        if pybrain_original:
            self.scaleLearningRate = self.covLearningRate
            self.batchSize = 4+int(floor(3*log(dim))) 
        else:
            self.scaleLearningRate = self.covLearningRate * 0.1
            self.batchSize = 10
        self.shapingFunction = HansenRanking()

        # some bookkeeping variables
        self._center = x_init
        self._A = eye(self.numParameters) # square root of covariance matrix
        self._invA = eye(self.numParameters)
        self._logDetA = 0.

    def step(self, fun):
        I = eye(self.numParameters)

        def produce_sample():
            cov = self._A.T.dot(self._A)
            print("cov : {0}".format(cov))
            samples = np.random.multivariate_normal(self._center, cov, self.batchSize)
            return samples

        samples = produce_sample()
        funevals = fun(samples)

        utilities = self.shapingFunction(funevals)
        utilities /= sum(utilities)  # make the utilities sum to 1
        if self.uniformBaseline:
            utilities -= 1./self.batchSize

        dCenter = dot(samples.T, utilities)
        covGradient = dot(array([outer(s,s) - I for s in samples]).T, utilities)
        covTrace = trace(covGradient)
        covGradient -= covTrace/self.numParameters * I
        dA = 0.5 * (self.scaleLearningRate * covTrace/self.numParameters * I
                    +self.covLearningRate * covGradient)

        self._lastLogDetA = self._logDetA
        self._lastInvA = self._invA

        self._center += self.centerLearningRate * dot(self._A, dCenter)
        self._A = dot(self._A, expm(dA))
        self._invA = dot(expm(-dA), self._invA)
        self._logDetA += 0.5 * self.scaleLearningRate * covTrace

if __name__=='__main__':
    xnes = XNES(array([2., 5.]))
    fun = lambda X: -np.sqrt(np.sum(X**2, axis=1))
    for i in range(1000):
        xnes.step(fun)
        print(xnes._center)




