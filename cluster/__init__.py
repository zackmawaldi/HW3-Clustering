"""
A Python module for clustering data using k-means, and scoring them using Silhouette scoring.
"""

from .kmeans import KMeans
from .silhouette import Silhouette
from .utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

