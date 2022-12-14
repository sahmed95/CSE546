import numpy as np

from utils import problem


@problem.tag("hw4-A")
def calculate_centers(
    data: np.ndarray, classifications: np.ndarray, num_centers: int
) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that calculates the centers given datapoints and their respective classifications/assignments.
    num_centers is additionally provided for speed-up purposes.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        classifications (np.ndarray): Array of shape (n,) full of integers in range {0, 1, ...,  num_centers - 1}.
            Data point at index i is assigned to classifications[i].
        num_centers (int): Number of centers for reference.
            Might be usefull for pre-allocating numpy array (Faster that appending to list).

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing new centers.
    """
    n, d =data.shape
    centers = np.zeros((num_centers, d))
    for i in range(num_centers):
        data_class = data[np.where(classifications ==i)]
        centers[i] = np.mean(data_class, axis = 0)
    return centers


@problem.tag("hw4-A")
def cluster_data(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Sub-routine of Lloyd's algorithm that clusters datapoints to centers given datapoints and centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.

    Returns:
        np.ndarray: Array of integers of shape (n,), with each entry being in range {0, 1, 2, ..., k - 1}.
            Entry j at index i should mean that j^th center is the closest to data[i] datapoint.
    """
    n = data.shape[0]
    k = centers.shape[0]
    cluster = np.zeros(n)
    for i in range(n):
        obj = np.zeros(k)
        for j in range(k): 
            val = data[i,:] - centers[j,:]
            obj[j] = np.linalg.norm(val)
        cluster[i] = np.argmin(obj)

    return cluster


@problem.tag("hw4-A")
def calculate_error(data: np.ndarray, centers: np.ndarray) -> float:
    """Calculates error/objective function on a provided dataset, with trained centers.

    Args:
        data (np.ndarray): Array of shape (n, d). Dataset to evaluate centers on.
        centers (np.ndarray): Array of shape (k, d). Each row is a center to which a datapoint can be clustered.
            These should be trained on training dataset.

    Returns:
        float: Single value representing mean objective function of centers on a provided dataset.
    """
    
    k = centers.shape[0]
    classifications = cluster_data(data, centers)
    err = 0
    n = data.shape[0]
    for i in range(k): 
        data_class = data[np.where(classifications ==i)]
        val = data_class-centers[i]
        inner = np.power(np.sum(np.power(val, 2),1), 0.5)
        err += np.sum(inner)
    return err/n
@problem.tag("hw4-A")
def lloyd_algorithm(
    data: np.ndarray, num_centers: int, epsilon: float = 10e-3
) -> np.ndarray:
    """Main part of Lloyd's Algorithm.

    Args:
        data (np.ndarray): Array of shape (n, d). Training data set.
        num_centers (int): Number of centers to train/cluster around.
        epsilon (float, optional): Epsilon for stopping condition.
            Training should stop when max(abs(centers - previous_centers)) is smaller or equal to epsilon.
            Defaults to 10e-3.

    Returns:
        np.ndarray: Array of shape (num_centers, d) containing trained centers.

    Note:
        - For initializing centers please use the first `num_centers` data points.
    """
    center = data[0:num_centers]
    diff = 100
    counter = 0
    while diff > epsilon:
        counter += 1
        clusters = cluster_data(data, center)
        new_center = calculate_centers(data,clusters,num_centers)
        inner = new_center-center
        diff = np.max(np.abs(inner))
        center = new_center
        print("counter:", counter)
        print("diff:", diff)

    return center
