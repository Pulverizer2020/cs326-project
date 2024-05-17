import numpy as np
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt


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


def als_matrix_factorization(ratings: np.array):
    train = np.zeros(ratings.shape)
    test = ratings.copy()

    # make random choice of testing data
    # training data will be randomly masked, making values into np.NaN
    for user_index in range(ratings.shape[0]):
        test_index = np.random.choice(
            np.flatnonzero(ratings[user_index]), size = 10, replace = False)

        train[user_index, test_index] = np.NaN()
        test[user_index, test_index] = ratings[user_index, test_index]
        
    # assert that training and testing set are truly disjoint
    assert np.all(train * test == 0)


    als = ExplicitMF(n_iters = 100, n_factors = 40, reg = 0.01)
    als.fit(train, test)

    plot_learning_curve(als)


    # return train, test

def plot_learning_curve(model):
    """visualize the training/testing loss"""
    linewidth = 3
    plt.plot(model.test_mse_record, label = 'Test', linewidth = linewidth)
    plt.plot(model.train_mse_record, label = 'Train', linewidth = linewidth)
    plt.xlabel('iterations')
    plt.ylabel('MSE')
    plt.legend(loc = 'best')


class ExplicitMF:
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

    def __init__(self, n_iters, n_factors, reg):
        self.reg = reg
        self.n_iters = n_iters
        self.n_factors = n_factors  
        
    def fit(self, train, test):
        """
        pass in training and testing at the same time to record
        model convergence, assuming both dataset is in the form
        of User x Item matrix with cells as ratings
        """
        self.n_user, self.n_item = train.shape
        self.user_factors = np.random.random((self.n_user, self.n_factors))
        self.item_factors = np.random.random((self.n_item, self.n_factors))
        
        # record the training and testing mse for every iteration
        # to show convergence later (usually, not worth it for production)
        self.test_mse_record  = []
        self.train_mse_record = []   
        for _ in range(self.n_iters):
            self.user_factors = self._als_step(train, self.user_factors, self.item_factors)
            self.item_factors = self._als_step(train.T, self.item_factors, self.user_factors) 
            predictions = self.predict()
            test_mse = self.compute_mse(test, predictions)
            train_mse = self.compute_mse(train, predictions)
            self.test_mse_record.append(test_mse)
            self.train_mse_record.append(train_mse)
        

        return self.user_factors.T @ self.item_factors    
    
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
        pred = self.user_factors.dot(self.item_factors.T)
        return pred
    
    @staticmethod
    def compute_mse(y_true, y_pred):
        """ignore zero terms prior to comparing the mse"""
        mask = np.nonzero(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        return mse
    
