from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform


class KMeans:
    def __init__(self, n_cl: int, n_init: int = 1,
                 initial_centers: Optional[np.ndarray] = None,
                 verbose: bool = False) -> None:
        """
        Parameters
        ----------
        n_cl: int
            number of clusters.
        n_init: int
            number of time the k-means algorithm will be run.
        initial_centers:
            If an ndarray is passed, it should be of shape (n_clusters, n_features)
            and gives the initial centers.
        verbose: bool
            whether or not to plot assignment at each iteration (default is True).
        """

        self.n_cl = n_cl
        self.n_init = n_init
        self.initial_centers = initial_centers
        self.verbose = verbose

    def _init_centers(self, X: np.ndarray, use_samples: bool = False):

        n_samples, dim = X.shape

        if use_samples:
            return X[np.random.choice(n_samples, size=self.n_cl)]

        centers = np.zeros((self.n_cl, dim))
        for i in range(dim):
            min_f, max_f = np.min(X[:, i]), np.max(X[:, i])
            centers[:, i] = uniform(low=min_f, high=max_f, size=self.n_cl)
        return centers

    def single_fit_predict(self, X: np.ndarray):
        """
        Kmeans algorithm.

        Parameters
        ----------
        X: ndarray
            data to partition, Expected shape (n_samples, dim).

        Returns
        -------
        centers: ndarray
            computed centers. Expected shape (n_cl, dim)

        assignment: ndarray
            computed assignment. Expected shape (n_samples,)
        """

        n_samples, dim = X.shape

        # initialize centers
        centers = np.array(self._init_centers(X)) if self.initial_centers is None \
            else np.array(self.initial_centers)

        old_assignments = np.zeros(shape=n_samples)

        if self.verbose:
            fig, ax = plt.subplots()

        while True:  # stopping criterion

            if self.verbose:
                ax.scatter(X[:, 0], X[:, 1], c=old_assignments, s=40)
                ax.plot(centers[:, 0], centers[:, 1], 'r*', markersize=20)
                ax.axis('off')
                plt.pause(1)
                plt.cla()

            """
            For each point in X compute the new assigment, based on the distances
            between it and all centers.
            """
            new_assignments = None

            """
            Update the centers.
            """
            centers = None

            """
            Check the break condition.
            """
            if True:
                break

            # update
            old_assignments = new_assignments

        if self.verbose:
            plt.close()

        return np.random.rand((self.n_cl, dim)), np.random.randint(0, self.n_cl - 1, X.shape[0])

    def compute_cost_function(self, X: np.ndarray, centers: np.ndarray,
                              assignments: np.ndarray):
        """
        Returns
        -------
        cost: float
            computed cost function.
        """

        return np.random.rand(1)[0]

    def fit_predict(self, X: np.ndarray):
        """
        Returns
        -------
        assignment: ndarray
            computed assignment. Expected shape (n_samples,)
        """

        for i in range(self.n_init):
            """
            1) Compute a single fit-and-predict step
            2) Evaluate the cost function and print it.
            3) If the cost function results lower than the current minimum,
                set the last solution as the current best one.
            """

        return np.random.randint(0, self.n_cl - 1, X.shape[0])
