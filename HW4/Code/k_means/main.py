if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    You should:
        a. Run Lloyd's Algorithm for k=10, and report 10 centers returned.
        b. For ks: 2, 4, 8, 16, 32, 64 run Lloyd's Algorithm,
            and report objective function value on both training set and test set.
            (All one plot, 2 lines)

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")
    
    # Part a : Lloyd's algorithm for k = 10
    k = 10 
    center_a = lloyd_algorithm(x_train, k) 
    fig, axes = plt.subplots(1,10)
    fig.suptitle('Clusters for k = 10')
    for i, ax in enumerate(axes):
        ax.imshow(center_a[i,:].reshape((28,28)), cmap = 'Greys')
        ax.set_title('k={}'.format(i+1))
        ax.axis('off')
    plt.show()
    
    # Part b 
    
    k = (2,4,8,16,32,64)
    

    train_obj = np.zeros(len(k))
    test_obj = np.zeros(len(k))
    for i in range(len(k)): 
        print("k:", k[i])
        centers = lloyd_algorithm(x_train, k[i])
        train_obj[i] += calculate_error(x_train, centers)
        test_obj[i] += calculate_error(x_test, centers)
        print("Train:", train_obj[i])
        print("Test:", test_obj[i])

    plt.figure()
    plt.plot(k, train_obj, 'rx-', label = 'Training')
    plt.plot(k, test_obj, 'bo-', label = 'Test')
    plt.title('Training and Test Error')
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
