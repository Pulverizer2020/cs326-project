import numpy as np
from sklearn.metrics import mean_squared_error


def scales_mean_matrix_factorization(ratings: np.array):
    # determine the midpoint of the scale (avg of min and max)
    
    scale_min = np.nanmin(ratings)
    scale_max = np.nanmax(ratings)
    scale_mid = (scale_max + scale_min) / 2

    # all unknown values are 0
    # all known values are shifted by scale_mid
    ratings = np.nan_to_num(ratings, nan=scale_mid)

    print("ratings", ratings)
    ratings = ratings - scale_mid

    return ratings.copy()


class ALSMatrixFactorization:
    """
    Train a matrix factorization model using Alternating Least Squares
    to predict empty entries in a matrix
    
    Parameters
    ----------
    n_iters : int
        number of iterations to train the algorithm
        
    n_factors : int
        number of latent factors to use in matrix 
        factorization model, some machine-learning libraries
        denote this as rank
        
    reg : float
        regularization term for item/user latent factors,
        since lambda is a keyword in python we use reg instead
    """

    def __init__(self, n_iters, n_factors, reg, train):
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.reg = reg  
        self.train = train
        
    def fit(self):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both datasets is in the form
        of User x Item matrix with cells as ratings
        """

        self.n_user, self.n_item = self.train.shape
        self.user_factors = np.random.random((self.n_user, self.n_factors))
        self.item_factors = np.random.random((self.n_item, self.n_factors))
        

        for _ in range(self.n_iters):
            self.user_factors = self._als_step(self.train, self.user_factors, self.item_factors)
            self.item_factors = self._als_step(self.train.T, self.item_factors, self.user_factors) 
        

        return self.user_factors @ self.item_factors.T
    
    def _als_step(self, ratings, solve_vecs, fixed_vecs):
        """
        when updating the user matrix,
        the item matrix is the fixed vector and vice versa
        """
        A = fixed_vecs.T.dot(fixed_vecs) + np.eye(self.n_factors) * self.reg
        b = ratings.dot(fixed_vecs)
        A_inv = np.linalg.inv(A)
        solve_vecs = b.dot(A_inv)
        return solve_vecs
    
    def predict(self):
        """predict ratings for every user and item"""
        pred = self.user_factors @ self.item_factors.T
        return pred
    
    def get_training_mse(self):
        """calculate the mean square error between known rating values and their values in the reconstructed matrix"""
        known_values = np.where(self.train != 0, True, False)
        mse = ((self.user_factors @ self.item_factors.T)[known_values] - self.train[known_values]) ** 2
        return np.average(mse)
    
    def get_training_mape(self):
        """calculate the mean absolute percentage error between known rating values and their values in the reconstructed matrix"""
        known_values = np.where(self.train != 0, True, False)
        mape = np.abs((self.user_factors @ self.item_factors.T)[known_values] - self.train[known_values]) / self.train[known_values]
        return np.average(mape) * 100

