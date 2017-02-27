#this is the protocol for storing the bayesian optimizer object
from gaussian_process_regressor_ import GaussianProcessRegressor_
import dill
from scipy.optimize import minimize
from numpy import array
from random import random
from numpy import exp, pi, sqrt
from scipy.special import erf
from halton import Halton

class BayesianOptimizer:
    #if a previous bayesian optimzer was made it can be loaded
    #else we do not do anything
    def __init__(self, filename = None):
        return None  
    
    #establish the regressor        
    def InitalizeRegressor(self, nu = 1.5, white_noise=True):
        self.gaussian_process_regressor_ = GaussianProcessRegressor_(nu=nu, white_noise=white_noise)
        return None   
    
    #set the various simulated annealing parameters    
    def InitializeOptimizer(self, rand_starts=100, tol=1e-10):
        self.rand_starts_ = rand_starts
        self.tol_ = tol
        return None
    
    #set the various suggestion related parameters
    def InitializeSuggestionEngine(self, num_suggestions=1, method='ei', rand_starts__max_pred_fitness=200):
        self.num_suggestions_ = num_suggestions
        self.method_ = method
        self.rand_starts__max_pred_fitness_ = rand_starts__max_pred_fitness
        
    #sets the maximum fitness value based on either standard or knowledge gradient
    def SetMaxFitnessEI(self):
        self.y_max_real_and_virtual_ = max(self.y_ + self.y_virtual_)
        return None
    
    #establish a library of initial samples to seed the model with
    def InitialSamples(self, features, fitness):
        self.X_ = features
        self.y_ = fitness
        self.X_virtual_ = []
        self.y_virtual_ = []
        return None
    
    #adds samples - typically this is a result of doing new experiments
    def AddSamples(self, features, fitness):
        self.X_ = self.X_ + features
        self.y_ = self.y_ + fitness
        return None
    
    #this controls the range of valid possible values for each axis
    def SetRanges(self, ranges):
        self.ranges_ = ranges
        self.halton_opt = Halton(self.ranges_)
        self.halton_kg = Halton(self.ranges_)
        return None
    
    #simple method for getting suggestions for some initial number of samples
    def SuggestInitialSamples(self, num_samples):
        halton = Halton(self.ranges_)
        return [halton.Get() for i in range(num_samples)]
    
    #adds some virtual samples - this is relevant to multipoint optimization
    def AddVirtualSamples(self, features, fitness):
        self.X_virtual_ = self.X_virtual_ + features
        self.y_virtual_ = self.y_virtual_ + fitness
        return None
    
    #clears out the virtual samples which happens at the beginning of any optimization
    def ClearVirtualSamples(self):
        self.X_virtual_ = []
        self.y_virtual_ = []
        return None
    
    #the expected improvement landscape to optimize and find the best of
    def ExpectedImprovement(self, x):
        f_mean, f_stdev  = self.gaussian_process_regressor_.predict(array([x]), return_std = True)
        f_mean = f_mean[0]
        f_stdev = f_stdev[0]
        f_max = self.y_max_real_and_virtual_
        f1 = f_stdev*exp(-((f_max - f_mean)**2.0)/(2.0*f_stdev**2.0))/sqrt(2.0*pi)
        f2 = (f_mean - f_max)/2.0
        f3 = (f_max - f_mean)*erf((f_max - f_mean)/(sqrt(2.0)*f_stdev))/2.0
        f = f1 + f2 + f3
        return f    
    
    #just the negative of the expected improvement
    def NegativeExpectedImprovement(self, x):
        return -1.0*self.ExpectedImprovement(x)
    
    #provides the expected fitness
    def ExpectedFitness(self, x):
        return self.gaussian_process_regressor_.predict(array([x]), return_std = False)[0]
    
    #provides the negative expected fitness
    def NegativeExpectedFitness(self, x):
        return -1.0*self.ExpectedFitness(x)
    
    #this searches the model space for the best predicted point to be used in knowledge gradient
    def SetMaxFitnessKG(self):
        #start with the best in the current set
        expected_fitness_best = max(self.y_ + self.y_virtual_)
        #print 'best in set %f' % expected_fitness_best
        #see if there is a better predicted value
        for iteration in range(0, self.rand_starts__max_pred_fitness_):
            #generate random initial guess and search
            #x0 = array([uniform(lower, upper) for lower, upper in self.ranges_])
            x0 = array(self.halton_kg.Get())
            result = minimize(self.NegativeExpectedFitness, x0, 
                              method='L-BFGS-B', tol=self.tol_, bounds=self.ranges_, options={'disp': True})
            expected_fitness = self.ExpectedFitness(result.x)
            expected_fitness_best = max(expected_fitness, expected_fitness_best)
        #now set based on what has been found as well as in the actual training dataset
        #print 'best overall found %f' % expected_fitness_best
        self.y_max_real_and_virtual_ = expected_fitness_best
        return None
            
    #wrapper for the scipy l-bfgs-b optimizer
    #perform some number of random initializations and take the best result    
    def Optimize(self):
        #set the max fitness according to chosen method
        if self.method_ == 'ei':
            self.SetMaxFitnessEI()
        elif self.method_ == 'kg':
            self.SetMaxFitnessKG()        
        #perform the optimization
        rand_starts = max(1, self.rand_starts_)
        ei_x__best = (0.0, None)
        for iteration in range(self.rand_starts_):
            #generate a random initial guess
            #x0 = array([uniform(lower, upper) for lower, upper in self.ranges_])
            x0 = array(self.halton_opt.Get())
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
    def Suggest(self):    
        suggestions = []
        #always clear out any virtual samples on first go
        self.ClearVirtualSamples()
        self.gaussian_process_regressor_.fit(self.X_ + self.X_virtual_, self.y_ + self.y_virtual_)
        f, x = self.Optimize()
        suggestions.append(x.tolist())
        self.AddVirtualSamples([x.tolist()], [self.ExpectedFitness(x)])
        #for multipoint we continue on with virtual samples included
        for i in range(self.num_suggestions_ - 1):
            self.gaussian_process_regressor_.fit(self.X_ + self.X_virtual_, self.y_ + self.y_virtual_)
            f, x = self.Optimize()
            suggestions.append(x.tolist())
            self.AddVirtualSamples([x.tolist()], [self.ExpectedFitness(x)])
        return suggestions