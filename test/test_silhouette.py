import pytest
import numpy as np

from cluster import KMeans, utils
from cluster.silhouette import Silhouette

from sklearn.metrics import silhouette_score


def test_score_correctness():
    # test score agaisnt known true values
    # do the testing for k 2 -> 50.

    silhouette = Silhouette()
    for k in range(2, 50):
        data, labels = utils.make_clusters(k=k)
        
        kmeans  = KMeans(k)
        kmeans.fit(data)
        predicted = kmeans.predict(data)

        my_scores = silhouette.score(X=data, y=predicted).mean()
        true_scores = silhouette_score(data, predicted)

        # check the difference is within acceptable floating / rounding shenanigans
        assert abs(my_scores - true_scores) < 0.01


def test_score_edge_cases():
    silhouette = Silhouette()

    # empty check
    with pytest.raises(ValueError):
        silhouette.score(np.array([]), np.array([]))

    # wrong dimentions
    with pytest.raises(ValueError):
        silhouette.score(np.array([1, 2, 3]), np.array([1, 2, 3]))
    
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        silhouette.score(X, y)

    # test single cluster
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 0])
    with pytest.raises(ValueError):
        silhouette.score(X, y)