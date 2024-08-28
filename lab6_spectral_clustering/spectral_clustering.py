import numpy as np
from datasets import two_moon_dataset, gaussians_dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.ion()

def spectral_clustering(data, n_cl, sigma=1., fiedler_solution = True): #fiedler_solution=False prende tutti gli autovalori
    """
    Spectral clustering.

    Parameters
    ----------
    data: ndarray
        data to partition, has shape (n_samples, dimensionality).
    n_cl: int
        number of clusters.
    sigma: float
        std of radial basis function kernel.
    fiedler_solution: bool
        return fiedler solution instead of kmeans

    Returns
    -------
    ndarray
        computed assignment. Has shape (n_samples,)
    """
    # compute affinity matrix
    #invece di fare ciclo for per sommare le distanze di tutti i punti
    #uso broadcasting fra due matrici di dati espande rispetto la dim 1 e la dim 2 e sommare due vettori di dim diverse
    Mij = ((np.expand_dims(data, 0)  -np.expand_dims(data, 1))**2).sum( 2 )
    affinity_matrix = np.exp( -(Mij / sigma ** 2) )
    
    # compute degree matrix
    affinity_matrix_diag = affinity_matrix.sum(1)
    degree_matrix = np.eye(affinity_matrix_diag.shape[0])*affinity_matrix_diag

    # compute laplacian
    laplacian_matrix = degree_matrix - affinity_matrix_diag 
    
    # compute eigenvalues and vectors (suggestion: np.linalg is your friend)
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix) 
    # v[:,i](tutta la colonna) is the eigenvector corresponding to the eigenvalue w[i].

    # ensure we are not using complex numbers - you shouldn't btw
    if eigenvalues.dtype == 'complex128':
        print("My dude, you got complex eigenvalues. Now I am not gonna break down, but you should totally give me higher sigmas (σ). (;")
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

    # sort eigenvalues and vectors
    indexes = eigenvalues.argsort()
    eigenvalues = eigenvalues[indexes]
    eigenvectors = eigenvectors[:, indexes]

    # SOLUTION A: Fiedler-vector solution -> è mincut, fa solo bipartizione-> prendo 2° autovalore 
    # - consider only the SECOND smallest eigenvector
    # - threshold it at zero
    # - return as labels
    labels = eigenvectors[: , 1 ] > 0
    if fiedler_solution:
        return labels

    # SOLUTION B: K-Means solution
    # - consider eigenvectors up to the n_cl-th 
    # - use them as features instead of data for KMeans
    # - You want to use sklearn's implementation (;
    # - return KMeans' clusters
    new_features = eigenvectors[: , :n_cl]
    labels = KMeans(n_cl).fit_predict(new_features) #la similarità delle new feat è la distanza con cui viene fatto il k-means

    return labels


def main_spectral_clustering():
    """
    Main function for spectral clustering.
    """

    # generate the dataset
    data, cl = two_moon_dataset(n_samples=300, noise=0.1)
    # data, cl = gaussians_dataset(n_gaussian=3, n_points=[100, 100, 70], mus=[[1,1], [-4,6], [8,8]], stds=[[1,1], [3,3],[1,1]])

    # visualize the dataset
    _, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)
    # plt.waitforbuttonpress()

    # run spectral clustering - tune n_cl and sigma!!!
    labels = spectral_clustering(data, n_cl=2, sigma=0.55)

    # visualize results
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main_spectral_clustering()
