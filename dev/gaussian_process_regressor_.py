from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import WhiteKernel
from numpy import array

class GaussianProcessRegressor_(BaseEstimator, ClassifierMixin):
    #standard init function
    def __init__(self, nu = 1.5):
        self.nu = nu
        return None
    #creates an explicit pipeline so that special features of the gp regressor can be accessed
    def fit(self, X, y, dy = None):
        #transformers
        self.standard_scaler_ = StandardScaler()
        self.standard_scaler_y_ = StandardScaler()
        #predictor
        dimensions = len(X[0])
        kernel = (Matern(length_scale = [1.0 for axis in range(dimensions)], nu = self.nu) 
                  + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-05, 100000.0)))
        if dy is None:
            self.gaussian_process_ = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10)
        else:
            self.gaussian_process_ = GaussianProcessRegressor(kernel = kernel, alpha = dy, n_restarts_optimizer = 10)
        #transform and fit
        X = self.standard_scaler_.fit_transform(X)
        y = array(y)
        y.shape = (len(y), 1)
        y = self.standard_scaler_y_.fit_transform(y)
        y.shape = (1, len(y))
        y = y[0]
        self.gaussian_process_.fit(X, y)
        return self
    #predict the thing and inverse transform the predicted scaled output  
    def predict(self, X, y = None, return_std = False):
        X = self.standard_scaler_.transform(X)
        y_predict_scaled = self.gaussian_process_.predict(X)
        if not return_std:
            y_predict_scaled = self.gaussian_process_.predict(X)
            y_predict = self.standard_scaler_y_.inverse_transform(y_predict_scaled)
            return y_predict
        else:
            y_predict_scaled, y_std_scaled = self.gaussian_process_.predict(X, return_std = True)
            y_predict = self.standard_scaler_y_.inverse_transform(y_predict_scaled)
            y_upper = self.standard_scaler_y_.inverse_transform(y_predict_scaled + y_std_scaled)
            return y_predict, (y_upper - y_predict)