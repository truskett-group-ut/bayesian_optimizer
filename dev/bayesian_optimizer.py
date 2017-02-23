#this is the protocol for storing the bayesian optimizer object
from gaussian_process_regressor_ import GaussianProcessRegressor_
import dill
#from scipy.optimize import anneal
from scipy.optimize import minimize
from numpy import array
from random import random
from numpy import exp, pi, sqrt
from scipy.special import erf
from random import uniform

class BayesianOptimizer:
    #if a previous bayesian optimzer was made it can be loaded
    #else we do not do anything
    def __init__(self, filename = None):
        return None     
    #establish the regressor        
    def InitalizeRegressor(self, nu = 1.5):
        self.gaussian_process_regressor_ = GaussianProcessRegressor_(nu = nu)
        return None   
    #set the various simulated annealing parameters    
    def InitializeOptimizer(self, rand_starts=1000, tol=1e-10):
        self.rand_starts_ = rand_starts
        self.tol_ = tol
        None
    #sets the maximum fitness value based on either standard or knowledge gradient
    def SetMaxFitness(self):
        self.y_max_real_and_virtual_ = max(self.y_ + self.y_virtual_)
        return None
    #establish a library of initial samples to seed the model with
    def InitialSamples(self, features, fitness):
        self.X_ = features
        self.y_ = fitness
        self.X_virtual_ = []
        self.y_virtual_ = []
        self.SetMaxFitness()
        return None
    #adds samples - typically this is a result of doing new experiments
    def AddSamples(self, features, fitness):
        self.X_ = self.X_ + features
        self.y_ = self.y_ + fitness
        self.SetMaxFitness()
        return None
    #this controls the range of valid possible values for each axis
    def SetRanges(self, ranges):
        self.ranges_ = ranges
        return None
    #adds some virtual samples - this is relevant to multipoint optimization
    def AddVirtualSamples(self, features, fitness):
        self.X_virtual_ = self.X_virtual_ + features
        self.y_virtual_ = self.y_virtual_ + fitness
        self.SetMaxFitness()
        return None
    #the fitness landscape to optimize and find the best fitness
    def ExpectedImprovement(self, x):
        f_mean, f_stdev  = self.gaussian_process_regressor_.predict(array([x]), return_std = True)
        f_mean = f_mean[0]
        f_stdev = f_stdev[0]
        f_max = self.y_max_real_and_virtual_
        f1 = exp(-((f_max - f_mean)**2.0)/(2.0*f_stdev**2.0))/(2.0*pi)
        f2 = (f_mean - f_max)/(f_stdev*2.0*sqrt(2.0*pi))
        f3 = (f_max - f_mean)*erf((f_max - f_mean)/(sqrt(2.0)*f_stdev))/(2.0*sqrt(2.0*pi)*f_stdev)
        f = f1 + f2 + f3
        #print f
        return f    
    #just the negative of the expected improvement
    def NegativeExpectedImprovement(self, x):
        return -1.0*self.ExpectedImprovement(x)
    #wrapper for the scipy l-bfgs-b optimizer
    #perform some number of random initializations and take the best result    
    def Optimize(self):
        rand_starts = max(1, self.rand_starts_)
        ei_x__best = (0.0, None)
        for iteration in range(self.rand_starts_):
            #generate a random initial guess
            x0 = array([uniform(lower, upper) for lower, upper in self.ranges_])
            #optimize the expected improvement over the valid ranges along each axis
            result = minimize(self.NegativeExpectedImprovement, x0, 
                              method='L-BFGS-B', tol=self.tol_, bounds=self.ranges_, options={'disp': True})
            ei_x = (self.ExpectedImprovement(result.x), result.x)
            if ei_x[0] != ei_x__best[0]:
                ei_x__best = max(ei_x, ei_x__best)
            else:
                ei_x__best = ei_x
        return ei_x__best
    #optimize the fitness and find a good suggested point to try next
    def Suggest(self, num_suggestions = 1, method = 'kg'):      
        #train model
        self.X_virtual_ = []
        self.y_virtual_ = []
        self.gaussian_process_regressor_.fit(self.X_ + self.X_virtual_, self.y_ + self.y_virtual_)
        #use either knowledge gradient or standard approach
        f, x = self.Optimize()
        return x.tolist()
        
        #for multipoint we must add virtual samples and repeat
        #for i in range(num_suggestions - 1):
        #    
        #    x = Optimize(rand_starts=20)
        