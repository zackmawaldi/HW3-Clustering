import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        
        ''' Edge case checks '''
        if X.size == 0 or y.size == 0:
            raise ValueError(f'X or y can not be empty. Currently: X.size = {X.size} and y.size = {y.size}')

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError(f'X must be a 2D array and y must be a 1D array. Currently: X.ndim = {X.ndim} and y.ndim = {y.ndim}')
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'The number of columns in X must match the length of y. Currently: X.shape[0] = {X.shape[0]} and y.shape[0] = {y.shape[0]}')
        
        if len(set(y)) <= 1:
            raise ValueError(f"By definition, silhouette scoring is comparative to other clusters. Thus, can't compute for number of labels <= 1. Currently: len(set(y)) == {len(set(y))}")
        
        Si_list = []
        for n in range(X.shape[0]):
            n_point = X[ n, : ].reshape(1, -1) # slice data to get n'th data point, reshape to be 2D
            n_point_label = y[n]

            # find ai (mean dist of current point to all points in cluster)
            
            # make subset of X that has all values with n_point_label and excludes n'th entry
            mask = (y == n_point_label) & (np.arange(X.shape[0]) != n)
            in_cluster_points = X[mask]

            
            # first, check if in_cluster_points is empty
            if in_cluster_points.size == 0:
                ai = 0
            else:
                # mean along rows, which I think is what I want
                ai = cdist(n_point, in_cluster_points).mean(axis=1)
            

            # find bi (the max of all mean dist of current point to all points in their respective clusters
            potential_bi = []
            for k in set(y):
                # exclude the datapoint's label
                if k == n_point_label:
                    continue

                out_cluster_points = X[y == k]

                # skip if no points in out_cluster_points
                if out_cluster_points.size == 0:
                    continue
                
                working_bi = cdist(n_point , out_cluster_points).mean(axis=1)
                potential_bi.append(working_bi)
            
            # in case all bi's got skipped...
            if potential_bi:
                bi = min(potential_bi)
            else:
                bi = 0

            # this avoid RuntimeWarning error, I think
            if ai == 0 and bi == 0:
                nth_Si = 0
            else:
                nth_Si = (bi - ai) / max(ai, bi)

            Si_list.append(nth_Si)
        
        return np.array(Si_list)




