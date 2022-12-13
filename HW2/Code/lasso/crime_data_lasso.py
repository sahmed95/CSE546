if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    # Splitting data set into response and explanatory variables
    Y_train = df_train["ViolentCrimesPerPop"] # Return the column named ``foo''.
    X_train = df_train.drop("ViolentCrimesPerPop", axis = 1)
    Y_test = df_test["ViolentCrimesPerPop"]
    X_test= df_test.drop("ViolentCrimesPerPop", axis = 1)

    # Finding max lambda 
    obj = 2*np.matmul(X_train.values.T, Y_train.values-np.mean(Y_train.values))
    lam_max = np.max(abs(obj), axis = 0)
    print(lam_max)
    lam = lam_max

    # Initializing weight
    weight = np.zeros(X_train.shape[1]) 

    convergence_delta = 1e-4

    [weight_max, bias_max] = train(X_train.values, Y_train.values, lam, convergence_delta, weight)
    lambdas = [lam]
    nonzero_weight = [np.count_nonzero(weight_max)]
    weight = weight_max

    # For regularization path 

    index_1 = df_train.columns.get_loc("agePct12t29")
    index_2 = df_train.columns.get_loc("pctWSocSec")
    index_3 = df_train.columns.get_loc("pctUrban")
    index_4 = df_train.columns.get_loc("agePct65up")
    index_5 = df_train.columns.get_loc("householdsize")
    
    # # Initializing regularization paths 

    # # Since, the first column is outcomes we subtract one from index 

    reg_1 = [weight_max[index_1-1]]
    reg_2 = [weight_max[index_2-1]]
    reg_3 = [weight_max[index_3-1]]
    reg_4 = [weight_max[index_4-1]]
    reg_5 = [weight_max[index_5-1]]

    # Initializing squared error on the training and test set

    test_error = [np.average((Y_test.values-X_test.values@weight_max-bias_max)**2)]
    train_error = [np.average((Y_train.values-X_train.values@weight_max-bias_max)**2)]

    while lam/2 >= 0.01: 
        lam = lam/2
        [weight_new, bias] = train(X_train.values, Y_train.values, lam, convergence_delta, weight)
        lambdas = np.append(lambdas, lam)
        nonzero_weight = np.append(nonzero_weight, np.count_nonzero(weight_new))
        
        # Regularization path
        reg_1 = np.append(reg_1, weight_new[index_1-1])
        reg_2 = np.append(reg_2, weight_new[index_2-1])
        reg_3 = np.append(reg_3, weight_new[index_3-1])
        reg_4 = np.append(reg_4, weight_new[index_4-1])
        reg_5 = np.append(reg_5, weight_new[index_5-1])
    
        # Squared error on the training and test set 
        test_error = np.append(test_error, np.average((Y_test.values-X_test.values@weight_new-bias)**2))
        train_error = np.append(train_error, np.average((Y_train.values-X_train.values@weight_new-bias)**2))
   
    # A3 Part c)
    plt.figure()
    plt.plot(lambdas, nonzero_weight, 'r+-')
    plt.xscale('log')
    plt.title('Number of nonzero weights vs Lambda')
    plt.xlabel('Lambda (log)')
    plt.ylabel('Number of nonzero elements of w')
    plt.show()


    # A3 Part d)
    plt.figure()
    plt.xscale('log')
    plt.plot(lambdas, reg_1, '-o', label = 'agePct12t29')
    plt.plot(lambdas, reg_2,'-o', label = 'pctWSocSec' )
    plt.plot(lambdas, reg_3,'-o', label ='pctUrban')
    plt.plot(lambdas, reg_4,'-o', label = 'agePct65up')
    plt.plot(lambdas, reg_5,'-o', label ='householdsize')
    plt.title('Regularization Paths')
    plt.xlabel('Lambda (log)')
    plt.ylabel('Corresponding Weights')
    plt.legend()
    plt.show()

    # A3 Part e)

    plt.figure()
    plt.xscale('log')
    plt.plot(lambdas, train_error, '-o', label = 'Training Error')
    plt.plot(lambdas, test_error, '-o', label = 'Test Error')
    plt.title('Training and Testing Error vs Lambda')
    plt.xlabel('Lambda (log)')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

    # A3 Part f)
    lam = 30
    [weight, bias] = train(X_train.values, Y_train.values, lam, convergence_delta, weight_max)
    id_max = np.argmax(weight)
    id_min = np.argmin(weight)
    max_feature = df_train.columns[id_max+1]
    min_feature = df_train.columns[id_min+1]

    print("Feature with the largest Lasso coefficient (%s) is %s" %(weight[id_max],max_feature))
    print("Feature with the  most negative Lasso coefficient (%s) is %s" %(weight[id_min],min_feature))
if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
