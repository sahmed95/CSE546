if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)

@problem.tag("hw3-A")
class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = LinearLayer(2,2, RNG)
    def forward(self, inputs): 
        soft = SoftmaxLayer()
        x =self.linear0(inputs)
        return soft(x)

@problem.tag("hw3-A")
class Network1_Sig(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size, RNG)
        self.linear_1 = LinearLayer(hidden_size, 2, RNG)

    def forward(self, inputs): 
        x = self.linear_0(inputs)
        sig = SigmoidLayer()
        soft = SoftmaxLayer()
        x = sig(x)
        return soft(self.linear_1(x))

@problem.tag("hw3-A")
class Network1_Relu(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size, RNG)
        self.linear_1 = LinearLayer(hidden_size, 2, RNG)

    def forward(self, inputs): 
        x = self.linear_0(inputs)
        relu = ReLULayer()
        soft = SoftmaxLayer()
        x = relu(x)
        return soft(self.linear_1(x))

class Network2_Sig_Relu(nn.Module):
    def __init__(self, hidden_size_1: int, hidden_size_2):
        super().__init__()
        self.linear_0 = LinearLayer(2, hidden_size_1, RNG)
        self.linear_1 = LinearLayer(hidden_size_1, hidden_size_2, RNG)
        self.linear_2 = LinearLayer(hidden_size_2, 2, RNG)

    def forward(self, inputs): 
        x = self.linear_0(inputs)
        sig = SigmoidLayer()
        relu = ReLULayer()
        soft = SoftmaxLayer()
        x = sig(x)
        x = self.linear_1(x)
        x = relu(x)
        return soft(self.linear_2(x))

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
        soft = SoftmaxLayer()
        x =relu(x)
        x = self.linear_1(x)
        x = sig(x)
        return soft(self.linear_2(x))


@problem.tag("hw3-A")
def crossentropy_parameter_search(
    dataloader_train: DataLoader, dataloader_val: DataLoader
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
    NOTE: Each model should end with a Softmax layer due to CrossEntropyLossLayer requirement.

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
    hist_1= train(dataloader_train,linear, CrossEntropyLossLayer(), SGDOptimizer, dataloader_val)
    train_err.append(hist_1['train'])
    val_err.append(hist_1['val'])


    
    # Model 2
    hidden_size = 2
    network1_sig = Network1_Sig(hidden_size)
    hist_2 = train(dataloader_train, network1_sig, CrossEntropyLossLayer(), SGDOptimizer, dataloader_val)
    train_err.append(hist_2['train'])
    val_err.append(hist_2['val'])

    
    # Model 3
    hidden_size = 2
    network1_relu = Network1_Relu(hidden_size)
    hist_3 = train(dataloader_train, network1_relu, CrossEntropyLossLayer(), SGDOptimizer, dataloader_val)
    train_err.append(hist_3['train'])
    val_err.append(hist_3['val'])

    # Model 4
    hidden_size_1 = 2
    hidden_size_2 = 2
    network2_sig_relu = Network2_Sig_Relu(hidden_size_1, hidden_size_2)
    hist_4 = train(dataloader_train, network2_sig_relu, CrossEntropyLossLayer(), SGDOptimizer, dataloader_val)
    train_err.append(hist_4['train'])
    val_err.append(hist_4['val'])

    

    # Model 5
    hidden_size_1 = 2
    hidden_size_2 = 2
    network2_relu_sig = Network2_Relu_Sig(hidden_size_1, hidden_size_2)
    hist_5 = train(dataloader_train, network2_relu_sig, CrossEntropyLossLayer(), SGDOptimizer, dataloader_val)
    train_err.append(hist_5['train'])
    val_err.append(hist_5['val'])


   
    model_name = ['linear', 'network1_sig', 'network1_relu', 'network2_sig_relu', 'network2_relu_sig']
    model = [linear, network1_sig, network1_relu, network2_sig_relu, network2_relu_sig]
    history = {}
    
    
    
    for i in range(len(model_name)):
        history[model_name[i]]= {'train': train_err[i], 'val': val_err[i], 'model': model[i]}
    
    return(history)




@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    accuracy = 0
    for batch in dataloader:
        data, labels = batch
        data, labels = data, labels
        y_hat = model(data)
        y_pred = torch.argmax(y_hat,1)
        accuracy +=torch.sum(y_pred==labels).item()
        
    return accuracy/len(dataloader.dataset)


@problem.tag("hw3-A", start_line=21)
def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    ce_dataloader_train = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y)),
        batch_size=32,
        shuffle=True,
        generator=RNG,
    )
    ce_dataloader_val = DataLoader(
        TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val)),
        batch_size=32,
        shuffle=False,
    )
    ce_dataloader_test = DataLoader(
        TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test)),
        batch_size=32,
        shuffle=False,
    )

    ce_configs = crossentropy_parameter_search(ce_dataloader_train, ce_dataloader_val)

    # Finding the best architecture 

    val_1 = np.min(ce_configs['linear']['val'])
    val_2 = np.min(ce_configs['network1_sig']['val'])
    val_3 = np.min(ce_configs['network1_relu']['val'])
    val_4 = np.min(ce_configs['network2_sig_relu']['val'])
    val_5 = np.min(ce_configs['network2_relu_sig']['val'])
    valid = [val_1,val_2, val_3, val_4, val_5]
    lowest_val= np.argmin(valid)
    models = list(ce_configs.keys())
    mod_name= models[lowest_val]
    model = ce_configs[mod_name]['model']
    print("The best performing architecture is:", mod_name)

    # Plotting the training and validation loss

    eps = range(100)
    plt.plot(eps, ce_configs['linear']['train'], '--',label = "Training Loss of Linear Regression Model")
    plt.plot(eps, ce_configs['network1_sig']['train'], '--',label ="Training Loss of NN1_Sig", )
    plt.plot(eps, ce_configs['network1_relu']['train'], '--',label ="Training Loss of NN1_Relu")
    plt.plot(eps, ce_configs['network2_sig_relu']['train'],'--',label ="Training Loss of NN2_Sig_Relu")
    plt.plot(eps, ce_configs['network2_relu_sig']['train'],'--', label ="Training Loss of NN1_Relu_Sig")
    plt.plot(eps, ce_configs['linear']['val'],label = "Validation Loss of Linear Regression Model")
    plt.plot(eps, ce_configs['network1_sig']['val'], label ="Validation Loss of NN1_Sig")
    plt.plot(eps, ce_configs['network1_relu']['val'], label ="Validation Loss of NN1_Relu")
    plt.plot(eps, ce_configs['network2_sig_relu']['val'],label ="Validation Loss of NN2_Sig_Relu")
    plt.plot(eps, ce_configs['network2_sig_relu']['val'], label ="Validation Loss of NN1_Relu_Sig")
    plt.title('Cross Entropy Parameter Search')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(fontsize = 5)
    plt.show()

    # Evaluating on the test set 
    
    acc = accuracy_score(model, ce_dataloader_test)
    print("Accuracy:", acc)
    plot_model_guesses(ce_dataloader_test, model,title =mod_name)

if __name__ == "__main__":
    main()
