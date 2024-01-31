import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        k = self.k
        data_dim = mat.shape[1] # columns = dimentions
        k_matrix = np.random.rand(k , data_dim) # consider selecting random data points for initialization
        loss_over_time = []

        for _ in range(self.max_iter):

            distances = cdist(mat, k_matrix)

            nth_centroid = np.argmin(distances, axis=1) # row wise find lowest distance

            # find new centroids
            for i in range(k):
                current_k_datapoints = mat[nth_centroid == i]
                
                # check that it's not emprty, then average (new k)
                if len(current_k_datapoints) > 0:
                    k_matrix[i] = current_k_datapoints.mean(axis=0) # do mean column wise along all datapoints


            SSE_loss = 0
            for m, n in enumerate(nth_centroid):
                SSE_loss += distances[m][n]
            
            loss_over_time.append(SSE_loss)
        
        self.centroids = k_matrix
        self.loss_over_time = loss_over_time

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        centroids = self.centroids
        distances = cdist(mat, centroids)
        predictions = np.argmin(distances, axis=1)

        return predictions


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
