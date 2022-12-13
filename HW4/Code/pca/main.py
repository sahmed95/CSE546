from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def reconstruct_demean(uk: np.ndarray, demean_data: np.ndarray) -> np.ndarray:
    """Given a demeaned data, create a recontruction using eigenvectors provided by `uk`.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_vec (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        np.ndarray: Array of shape (n, d).
            Each row should correspond to row in demean_data,
            but first compressed and then reconstructed using uk eigenvectors.
    """
    
    eigv_sum = uk@uk.T
    return demean_data@eigv_sum

@problem.tag("hw4-A")
def reconstruction_error(uk: np.ndarray, demean_data: np.ndarray) -> float:
    """Given a demeaned data and some eigenvectors calculate the squared L-2 error that recontruction will incur.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        float: Squared L-2 error on reconstructed data.
    """
    rec = reconstruct_demean(uk, demean_data)
    diff = np.power(np.linalg.norm(rec-demean_data),2)
    err = np.sum(diff, axis= 0)
    return err/demean_data.shape[0]
    


@problem.tag("hw4-A")
def calculate_eigen(demean_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given demeaned data calculate eigenvalues and eigenvectors of it.

    Args:
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays representing:
            1. Eigenvalues array with shape (d,)
            2. Matrix with eigenvectors as columns with shape (d, d)
    """
    n = demean_data.shape[0]
    X = np.matmul(demean_data.T, demean_data)/n
    eig, eigv= np.linalg.eig(X)
    return (eig,eigv)
   


@problem.tag("hw4-A", start_line=2)
def main():
    """
    Main function of PCA problem. It should load data, calculate eigenvalues/-vectors,
    and then answer all questions from problem statement.

    Part A:
        - Report 1st, 2nd, 10th, 30th and 50th largest eigenvalues
        - Report sum of eigenvalues

    Part C:
        - Plot reconstruction error as a function of k (# of eigenvectors used)
            Use k from 1 to 101.
            Plot should have two lines, one for train, one for test.
        - Plot ratio of sum of eigenvalues remaining after k^th eigenvalue with respect to whole sum of eigenvalues.
            Use k from 1 to 101.

    Part D:
        - Visualize 10 first eigenvectors as 28x28 grayscale images.

    Part E:
        - For each of digits 2, 6, 7 plot original image, and images reconstruced from PCA with
            k values of 5, 15, 40, 100.
    """
    (x_tr, y_tr), (x_test, _) = load_dataset("mnist")
    mu = np.mean(x_tr, axis = 0)
    mu_test = np.mean(x_test, axis = 0)
    x = x_tr - mu
    x_t = x_test-mu_test
    
    # Part A

    (eig, eigv) = calculate_eigen(x)
    print("The 1st eigenvalue is:", eig[0])
    print("The 2nd eigenvalue is:", eig[1])
    print("The 10th eigenvalue is:", eig[9])
    print("The 30th eigenvalue is:", eig[29])
    print("The 50th eigenvalue is", eig[49])

    sum_eig = np.sum(eig)
    print("The sum of eigenvalues:", sum_eig)

    # Part C 

    k = np.linspace(1,100,100, dtype = int)
    err_train = np.zeros(len(k))
    err_test = np.zeros(len(k))
    ratio = np.zeros(len(k))
    for i in range(len(k)):
        print(i)
        ind = k[i]
        eigv_k = eigv[:,0:ind]
        eig_k = eig[0:ind]
        err_train[i] = reconstruction_error(eigv_k, x)
        err_test[i] = reconstruction_error(eigv_k, x_t)
        ratio[i] = 1-(np.sum(eig_k)/sum_eig)

    plt.figure()
    plt.plot(k, err_train, 'rx-', label = 'Training')
    plt.plot(k, err_test, 'bx-', label = 'Test')
    plt.title('Reconstruction Error')
    plt.xlabel('Number of eigenvectors used (k)')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(k, ratio, 'r-')
    plt.title('Ratio of sum of eigenvalues remaining')
    plt.xlabel('Number of eigenvalues used (k)')
    plt.ylabel('Ratio')
    plt.show()

    # Part D

    fig, axes = plt.subplots(1,10)
    for i, ax in enumerate(axes):
        ax.imshow(eigv[:,i].reshape((28,28)), cmap = 'Greys')
        ax.set_title('Eigv #: {}'.format(i+1))
        ax.axis('off')
    plt.show()

    # Part E

    # Let us say that we will pick the 10th image of 2,6, 7 in our data

    k = [5, 15, 40, 100]
    (eig, eigv) = calculate_eigen(x)

    fig, axes = plt.subplots(3, 5)
    digits = [2,6,7]
    idx = [np.where(y_tr == digit)[0][9] for digit in digits]
    for i,ax in enumerate(axes):
        id = idx[i]
        for j in range(5):
            if j == 0:  
                axes[i,j].imshow(x_tr[id,:].reshape(28,28), cmap = 'Greys')
                axes[i,j].set_title('Original')
                axes[i,j].axis('off')
            else: 
                val = k[j-1]
                y_hat = reconstruct_demean(eigv[:,0:val],x)
                axes[i,j].imshow(y_hat[id].reshape(28,28), cmap = 'Greys')
                axes[i,j].set_title('k = {}'.format(val))
                axes[i,j].axis('off')
    plt.show()

    # k =[32, 64, 128]
    # digits = [0,1,2,3,4,5,6,7,8,9]
    # fig, axes = plt.subplots(10, 4)
    # (eig_k, eigv_k) = calculate_eigen(x)
    # idx = [np.where(y_tr == digit)[0][0] for digit in digits]
    # for i,ax in enumerate(axes):
    #     id = idx[i]
    #     for j in range(4):
    #         if j == 0:  
    #             axes[i,j].imshow(x_tr[id,:].reshape(28,28))
    #             axes[i,j].set_title('Original')
    #             axes[i,j].axis('off')
    #         else: 
    #             val = k[j-1]
    #             y_hat = reconstruct_demean(eigv_k[:,0:val],x)
    #             axes[i,j].imshow(y_hat[id].reshape(28,28))
    #             axes[i,j].set_title('k = {}'.format(val))
    #             axes[i,j].axis('off')
    # plt.show()
if __name__ == "__main__":
    main()
