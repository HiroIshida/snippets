# Natural Evolution Strategy, IJML (2014) algorithm 5

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, expm

def create_fax_unless_specifeid(fax):
    if fax is None:
        return plt.subplots()
    return fax

class Rosen:
    def __init__(self):
        pass

    def __call__(self, pts):
        if pts.ndim == 1:
            pts = np.array([pts])
        a = 2.0
        b = 100.0
        X, Y = pts[:, 0], pts[:, 1]
        #f = (a - X) ** 2 + b * (Y - X**2)**2
        f = X ** 2 + Y ** 2
        return f

    def show(self, fax=None):
        N = 100
        b_min = np.array([-3, -3])
        b_max = - b_min
        xlin, ylin = [np.linspace(b_min[i], b_max[i], N) for i in range(2)]
        X, Y = np.meshgrid(xlin, ylin)
        pts = np.array(list(zip(X.flatten(), Y.flatten())))
        Z = self.__call__(pts).reshape((N, N))
        fig, ax = create_fax_unless_specifeid(fax)
        #ax.contourf(X, Y, Z, levels=[2**n for n in range(17)])
        ax.contourf(X, Y, Z)

class NaturalEvolution:
    def __init__(self, x_init):
        self.x_mean = x_init
        self.n_dim = len(x_init)

        self.cov = np.diag((1, 1)) 
        A = np.linalg.cholesky(self.cov).T
        self.sigma = abs(np.linalg.det(A)) ** (1/self.n_dim)
        self.B = A/self.sigma

        #self.lam = int(4 + 3*np.log(self.n_dim)) 
        self.lam = 1000
        self.eta_sigma = 3*(3+np.log(self.n_dim))*(1.0/(5*self.n_dim*np.sqrt(self.n_dim))) 
        self.eta_bmat = 3*(3+np.log(self.n_dim))*(1.0/(5*self.n_dim*np.sqrt(self.n_dim)))
        self.eta_mu = 1.0

    def step(self, fun):
        s_lst = np.random.randn(self.lam, self.n_dim)
        z_lst = self.x_mean + self.sigma * np.dot(s_lst, self.B)
        f_lst = [fun(z).item() for z in z_lst]

        f_mean = np.mean(f_lst)

        rankebased = False
        if rankebased:
            idxes_upper = np.where(f_lst - f_mean < 0)[0]
            n_nonzero = len(idxes_upper)
            u_lst = 1.0/float(n_nonzero) * (f_lst - f_mean < 0) 

        u_lst = 1.0/self.lam * np.ones(self.lam)

        nabla_delta = sum([u * s for u, s in zip(u_lst, s_lst)])
        tmp_lst = [u * (np.outer(s, s) - np.eye(self.n_dim)) for s in s_lst]
        nabla_M = sum(tmp_lst)
        nabla_sigma = np.trace(nabla_M) / self.n_dim
        nabla_B = nabla_M  - nabla_sigma * np.eye(self.n_dim)

        self.x_mean += self.eta_mu * self.sigma * nabla_delta.dot(self.B.T)
        self.sigma *= np.exp(self.eta_sigma * 0.5 * nabla_sigma)
        self.B = self.B.dot(expm(self.eta_bmat * 0.5 * nabla_B))

class XNES(DistributionBasedOptimizer):
    """ NES with exponential parameter representation. """

    # parameters, which can be set but have a good (adapted) default value
    covLearningRate = None
    centerLearningRate = 1.0
    scaleLearningRate = None
    uniformBaseline = True
    batchSize = None
    shapingFunction = HansenRanking()
    importanceMixing = False
    forcedRefresh = 0.01

    # fixed settings
    mustMaximize = True
    storeAllEvaluations = True
    storeAllEvaluated = True
    storeAllDistributions = False

    def _additionalInit(self):
        # good heuristic default parameter settings
        dim = self.numParameters
        if self.covLearningRate is None:
            self.covLearningRate = 0.6*(3+log(dim))/dim/sqrt(dim)
        if self.scaleLearningRate is None:
            self.scaleLearningRate = self.covLearningRate
        if self.batchSize is None:
            if self.importanceMixing:
                self.batchSize = 10*dim
            else:
                self.batchSize = 4+int(floor(3*log(dim)))

        # some bookkeeping variables
        self._center = self._initEvaluable.copy()
        self._A = eye(self.numParameters) # square root of covariance matrix
        self._invA = eye(self.numParameters)
        self._logDetA = 0.
        self._allPointers = []
        self._allGenSteps = [0]
        if self.storeAllDistributions:
            self._allDistributions = [(self._center.copy(), self._A.copy())]

    def _learnStep(self):
        """ Main part of the algorithm. """
        I = eye(self.numParameters)
        self._produceSamples()
        utilities = self.shapingFunction(self._currentEvaluations)
        utilities /= sum(utilities)  # make the utilities sum to 1
        if self.uniformBaseline:
            utilities -= 1./self.batchSize
        samples = array(list(map(self._base2sample, self._population)))

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
        if self.storeAllDistributions:
            self._allDistributions.append((self._center.copy(), self._A.copy()))

    @property
    def _lastA(self): return self._allDistributions[-2][1]
    @property
    def _lastCenter(self): return self._allDistributions[-2][0]
    @property
    def _population(self):
        if self._wasUnwrapped:
            return [self._allEvaluated[i].params for i in self._pointers]
        else:
            return [self._allEvaluated[i] for i in self._pointers]

    @property
    def _currentEvaluations(self):
        fits = [self._allEvaluations[i] for i in self._pointers]
        if self._wasOpposed:
            fits = [-x for x in fits]
        return fits

    def _produceSample(self):
        return randn(self.numParameters)

    def _sample2base(self, sample):
        """ How does a sample look in the outside (base problem) coordinate system? """
        return dot(self._A, sample)+self._center

    def _base2oldsample(self, e):
        """ How would the point have looked in the previous reference coordinates? """
        return dot(self._lastInvA, (e - self._lastCenter))

    def _base2sample(self, e):
        """ How does the point look in the present one reference coordinates? """
        return dot(self._invA, (e - self._center))

    def _oldpdf(self, s):
        s = self._base2oldsample(self._sample2base(s))
        return exp(-0.5*dot(s,s)- self._lastLogDetA)

    def _newpdf(self, s):
        return exp(-0.5*dot(s,s)- self._logDetA)

    def _produceSamples(self):
        """ Append batch size new samples and evaluate them. """
        reuseindices = []
        if self.numLearningSteps == 0 or not self.importanceMixing:
            [self._oneEvaluation(self._sample2base(self._produceSample())) for _ in range(self.batchSize)]
            self._pointers = list(range(len(self._allEvaluated)-self.batchSize, len(self._allEvaluated)))
        else:
            reuseindices, newpoints = importanceMixing(list(map(self._base2sample, self._currentEvaluations)),
                                                       self._oldpdf, self._newpdf, self._produceSample, self.forcedRefresh)
            [self._oneEvaluation(self._sample2base(s)) for s in newpoints]
            self._pointers = ([self._pointers[i] for i in reuseindices]+
                              list(range(len(self._allEvaluated)-self.batchSize+len(reuseindices), len(self._allEvaluated))))
        self._allGenSteps.append(self._allGenSteps[-1]+self.batchSize-len(reuseindices))
        self._allPointers.append(self._pointers)

fun = Rosen()
fun.show()
#plt.show()
nes = NaturalEvolution(np.array([-1, 1.0]))
for i in range(1000):
    nes.step(fun)
    print(nes.x_mean)
    print(nes.sigma)
    #print(fun(nes.x_mean))

#fun.show()
#plt.show()
