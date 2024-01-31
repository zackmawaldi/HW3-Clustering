import pytest
import numpy as np
from cluster import KMeans, utils

def test_kmeans_init():
    # test valid initialization
    KMeans(k=3)

    # test bad k, tol, and max_iter initialization
    with pytest.raises(ValueError):
        KMeans(k=0)

    with pytest.raises(ValueError):
        KMeans(k=3, tol=-1)

    with pytest.raises(ValueError):
        KMeans(k=3, max_iter=0)


def test_kmeans_fit():
    # get valid dataset
    data, labels = utils.make_clusters()
    kmeans = KMeans(k=3)
    kmeans.fit(data)

    # test on empty matrix
    empty_array = np.array([])
    with pytest.raises(ValueError):
        kmeans.fit(empty_array)
    
    # test k > data count
    one_row_array = np.array([[0,0]])
    with pytest.raises(ValueError):
        kmeans.fit(one_row_array)


def test_kmeans_predict():
    # get valid dataset
    k = 3
    kmeans = KMeans(k=k)
    data, labels = utils.make_clusters()

    # test predicting before fitting
    with pytest.raises(ValueError):
        kmeans.predict(data)

    # test fit then predict
    kmeans.fit(data)
    new_data = np.random.uniform(low=-10, high=10, size=(10,2))
    predictions = kmeans.predict(new_data)

    assert predictions.shape == (10,)
    assert set(predictions).issubset( (0,1,2) )

    # test inputted wrong row count (number of features)
    new_data = np.random.rand(10, 3)
    with pytest.raises(ValueError):
        predictions = kmeans.predict(new_data)


def test_kmeans_get_error():
    kmeans = KMeans(k=3)
    data, labels = utils.make_clusters()

    # test get error without fitting
    with pytest.raises(LookupError):
        kmeans.get_error()

    # test get error after fitting
    kmeans.fit(data)
    error = kmeans.get_error()

    assert error >= 0


def test_kmeans_get_centroids():
    kmeans = KMeans(k=3)
    data, labels = utils.make_clusters()

    # test get centroids without fitting
    with pytest.raises(LookupError):
        kmeans.get_centroids()

    # test fiting then get centroids
    kmeans.fit(data)
    print(kmeans.get_centroids())
    assert kmeans.get_centroids().shape == (3, 2)