if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    accuracy = 0
    for batch in dataloader:
        data, labels = batch
        data, labels = data, labels
        y_hat = model(data)
        y_pred = torch.argmax(y_hat,1)
        y_labels = torch.argmax(labels,1)
        accuracy +=torch.sum(y_pred==y_labels).item()
        
    return accuracy/len(dataloader.dataset)

@problem.tag("hw3-A")
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = LinearLayer(2,2, RNG)
    def forward(self, inputs): 
        x =self.linear0(inputs)
        return x

@problem.tag("hw3-A")
class Network1_Sig(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size, RNG)
        self.linear_1 = LinearLayer(hidden_size, 2, RNG)

    def forward(self, inputs): 
        x = self.linear_0(inputs)
        sig = SigmoidLayer()
        x = sig(x)
        return self.linear_1(x)

@problem.tag("hw3-A")
class Network1_Relu(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size, RNG)
        self.linear_1 = LinearLayer(hidden_size, 2, RNG)

    def forward(self, inputs): 
        x = self.linear_0(inputs)
        relu = ReLULayer()
        x = relu(x)
        return self.linear_1(x)

class Network2_Sig_Relu(nn.Module):
    def __init__(self, hidden_size_1: int, hidden_size_2):
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size_1,RNG)
        self.linear_1 = LinearLayer(hidden_size_1, hidden_size_2, RNG)
        self.linear_2 = LinearLayer(hidden_size_2, 2, RNG)

    def forward(self, inputs): 
        x = self.linear_0(inputs)
        sig = SigmoidLayer()
        relu = ReLULayer()
        x = sig(x)
        x = self.linear_1(x)
        x = relu(x)
        return self.linear_2(x)

class Network2_Relu_Sig(nn.Module):
    def __init__(self, hidden_size_1: int, hidden_size_2): 
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size_1, RNG)
        self.linear_1 = LinearLayer(hidden_size_1, hidden_size_2, RNG)
        self.linear_2 = LinearLayer(hidden_size_2, 2, RNG)

    def forward(self, inputs): 
        x = self.linear_0(inputs)
        relu = ReLULayer()
        sig = SigmoidLayer()
        x =relu(x)
        x = self.linear_1(x)
        x = sig(x)
        return self.linear_2(x)

@problem.tag("hw3-A")
def mse_parameter_search(
    dataloader_train: DataLoader, dataloader_val: DataLoader
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Args:
        dataloader_train (DataLoader): Dataloader for training dataset.
        dataloader_val (DataLoader): Dataloader for validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    
    train_err= []
    val_err = []

    # Model 1

    linear = Linear()
    hist_1= train(dataloader_train,linear, MSELossLayer(), SGDOptimizer, dataloader_val)
    train_err.append(hist_1['train'])
    val_err.append(hist_1['val'])
    
    # Model 2

    hidden_size = 2
    network1_sig = Network1_Sig(hidden_size)
    hist_2 = train(dataloader_train, network1_sig, MSELossLayer(), SGDOptimizer, dataloader_val)
    train_err.append(hist_2['train'])
    val_err.append(hist_2['val'])

    # Model 3
    hidden_size = 2
    network1_relu = Network1_Relu(hidden_size)
    hist_3 = train(dataloader_train, network1_relu, MSELossLayer(), SGDOptimizer, dataloader_val)
    train_err.append(hist_3['train'])
    val_err.append(hist_3['val'])

    # Model 4

    hidden_size_1 = 2
    hidden_size_2 = 2
    network2_sig_relu = Network2_Sig_Relu(hidden_size_1, hidden_size_2)
    hist_4 = train(dataloader_train, network2_sig_relu, MSELossLayer(), SGDOptimizer, dataloader_val)
    train_err.append(hist_4['train'])
    val_err.append(hist_4['val'])

    # Model 5

    hidden_size_1 = 2
    hidden_size_2 = 2
    network2_relu_sig = Network2_Relu_Sig(hidden_size_1, hidden_size_2)
    hist_5 = train(dataloader_train, network2_relu_sig, MSELossLayer(), SGDOptimizer, dataloader_val)
    train_err.append(hist_5['train'])
    val_err.append(hist_5['val'])

    # Creating the dictionary

    model_name = ['linear', 'network1_sig', 'network1_relu', 'network2_sig_relu', 'network2_relu_sig']
    model = [linear, network1_sig, network1_relu, network2_sig_relu, network2_relu_sig]
    history = {}

    for i in range(len(model_name)):
        history[model_name[i]]= {'train': train_err[i], 'val': val_err[i], 'model': model[i]}
   
    return(history)

    

@problem.tag("hw3-A", start_line=21)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    mse_dataloader_train = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y))),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )
    mse_dataloader_val = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
        ),
        batch_size=32,
        shuffle=False,
    )
    mse_dataloader_test = DataLoader(
        TensorDataset(
            torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
        ),
        batch_size=32,
        shuffle=False,)
     
    hist = mse_parameter_search(mse_dataloader_train, mse_dataloader_val)

    # Finding the best architecture 

    val_1 = np.min(hist['linear']['val'])
    val_2 = np.min(hist['network1_sig']['val'])
    val_3 = np.min(hist['network1_relu']['val'])
    val_4 = np.min(hist['network2_sig_relu']['val'])
    val_5 = np.min(hist['network2_relu_sig']['val'])
    valid = [val_1,val_2, val_3, val_4, val_5]
    lowest_val= np.argmin(valid)
    models = list(hist.keys())
    mod_name= models[lowest_val]
    model = hist[mod_name]['model']

    print("The best performing architecture is:", mod_name)

    # Plotting the training and validation loss

    eps = range(100)
    plt.plot(eps, hist['linear']['train'], '--',label = "Training Loss of Linear Regression Model")
    plt.plot(eps, hist['network1_sig']['train'],'--', label ="Training Loss of NN1_Sig")
    plt.plot(eps, hist['network1_relu']['train'], '--', label ="Training Loss of NN1_Relu")
    plt.plot(eps, hist['network2_sig_relu']['train'], '--', label ="Training Loss of NN2_Sig_Relu")
    plt.plot(eps, hist['network2_relu_sig']['train'], '--', label ="Training Loss of NN1_Relu_Sig")
    plt.plot(eps, hist['linear']['val'],label = "Validation Loss of Linear Regression Model")
    plt.plot(eps, hist['network1_sig']['val'], label ="Validation Loss of NN1_Sig")
    plt.plot(eps, hist['network1_relu']['val'], label ="Validation Loss of NN1_Relu")
    plt.plot(eps, hist['network2_sig_relu']['val'],label ="Validation Loss of NN2_Sig_Relu")
    plt.plot(eps, hist['network2_sig_relu']['val'], label ="Validation Loss of NN1_Relu_Sig")
    plt.title('MSE Parameter Search')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

    # Evaluating on the test set 
    
    acc = accuracy_score(model, mse_dataloader_test)
    print("Accuracy:", acc)
    plot_model_guesses(mse_dataloader_test, model, title = mod_name)


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
