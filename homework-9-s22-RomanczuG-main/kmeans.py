import numpy as np
from sqlalchemy import false, true
from cluster import createClusters
from point import makePointList


def kmeans(point_data, cluster_data):
    """Performs k-means clustering on points.

    Args:
      point_data: a k-by-d numpy array used for creating a list of Points.
      cluster_data: A k-by-d numpy array used for creating a list of Clusters.

    Returns:
      A list of clusters (with update centers) after peforming k-means
      clustering on the points initialized from point_data
    """
    # Fill in
    myFlag = True

    # 1. Make list of points using makePointList and point_data
    points = makePointList(point_data)
    # 2. Make list of clusters using createClusters and cluster_data
    clusters = createClusters(cluster_data)
    # 3. As long as points keep moving:
    while(myFlag):
      myFlag = False
      for point in points:
        cluster = point.closest(clusters)
        myFlag = True if point.moveToCluster(cluster) else False
        
      #   A. Move every point to its closest cluster (use Point.closest and
      #     Point.moveToCluster)
      #     Hint: keep track here whether any point changed clusters by
      #           seeing if any moveToCluster call returns "True"
      for cluster in clusters:
        cluster.updateCenter()
    #   B. Update the centers for each cluster (use Cluster.updateCenter)

    # 4. Return the list of clusters, with the centers in their final positions
    return clusters


if __name__ == "__main__":
    data = np.array(
        [[0.5, 2.5], [0.3, 4.5], [-0.5, 3], [0, 1.2], [10, -5], [11, -4.5], [8, -3]],
        dtype=float,
    )
    centers = np.array([[0, 0], [1, 1]], dtype=float)

    clusters = kmeans(data, centers)
    for c in clusters:
        c.printAllPoints()
