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
        Si_list = []
        for n in range(X.shape[0]):
            n_point = X[ n, : ].reshape(1, -1) # slice data to get n'th data point, reshape to be 2D
            n_point_label = y[n]

            # find ai (mean dist of current point to all points in cluster)
            in_cluster_points = X[y == n_point_label]
            
            # delete working from list of distances to make, by column wise search
            in_cluster_points = np.delete(in_cluster_points, np.where(in_cluster_points == n_point), axis=0)

            # mean along rows, which I think is what I want
            ai = cdist(n_point , in_cluster_points).mean(axis=1)

            # find bi (the max of all mean dist of current point to all points in their respective clusters
            potential_bi = []
            for k in set(y):
                # exclude the datapoint's label
                if k == n_point_label:
                    continue

                out_cluster_points = X[y == k]
                
                working_bi = cdist(n_point , out_cluster_points).mean(axis=1)
                potential_bi.append(working_bi)
            
            bi = min(potential_bi)

            nth_Si = (bi - ai) / max(ai, bi)

            Si_list.append(nth_Si)
        
        return np.array(Si_list)




