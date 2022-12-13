import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


@problem.tag("hw4-A")
def F1(h: int) -> nn.Module:
    """Model F1, it should performs an operation W_d * W_e * x as written in spec.

    Note:
        - While bias is not mentioned explicitly in equations above, it should be used.
            It is used by default in nn.Linear which you can use in this problem.

    Args:
        h (int): Dimensionality of the encoding (the hidden layer).

    Returns:
        nn.Module: An initialized autoencoder model that matches spec with specific h.
    """
    
    d = 784
    linear_0 = nn.Linear(d,h)
    linear_1 = nn.Linear(h,d)
    return torch.nn.Sequential(linear_0, linear_1)
    




@problem.tag("hw4-A")
def F2(h: int) -> nn.Module:
    """Model F1, it should performs an operation ReLU(W_d * ReLU(W_e * x)) as written in spec.

    Note:
        - While bias is not mentioned explicitly in equations above, it should be used.
            It is used by default in nn.Linear which you can use in this problem.

    Args:
        h (int): Dimensionality of the encoding (the hidden layer).

    Returns:
        nn.Module: An initialized autoencoder model that matches spec with specific h.
    """
    
    d = 784 
    linear_0 = nn.Linear(d,h)
    linear_1 = nn.Linear(h,d)
    return torch.nn.Sequential(linear_0, nn.ReLU(), linear_1, nn.ReLU())


@problem.tag("hw4-A")
def train(
    model: nn.Module, optimizer: Adam, train_loader: DataLoader, epochs: int = 40
) -> float:
    """
    Train a model until convergence on train set, and return a mean squared error loss on the last epoch.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
            Hint: You can try using learning rate of 5e-5.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce x
            where x is FloatTensor of shape (n, d).

    Note:
        - Unfortunately due to how DataLoader class is implemented in PyTorch
            "for x_batch in train_loader:" will not work. Use:
            "for (x_batch,) in train_loader:" instead.

    Returns:
        float: Final training error/loss
    """
    lr = 5e-5
    optimize= optimizer(model.parameters(), lr)
    loss_fn = nn.MSELoss()
    for i in range(epochs): 
        loss_epoch = 0
        for (batch,) in train_loader: 
            batch = batch.view(-1, 784)
            optimize.zero_grad()
            y_hat = model(batch)
            loss = loss_fn(y_hat, batch) 
            loss_epoch += loss.item()
            loss.backward()
            optimize.step()
    return loss_epoch/len(train_loader)



@problem.tag("hw4-A")
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Evaluates a model on a provided dataset.
    It should return an average loss of that dataset.

    Args:
        model (Module): TRAINED Model to evaluate. Either F1, or F2 in this problem.
        loader (DataLoader): DataLoader with some data.
            You can iterate over it like a list, and it will produce x
            where x is FloatTensor of shape (n, d).

    Returns:
        float: Mean Squared Error on the provided dataset.
    """
    loss = 0
    loss_fn = nn.MSELoss()
    for (batch_test,) in loader:
        batch_test= batch_test.view(-1, 784)
        y_hat = model(batch_test)
        losses = loss_fn(y_hat, batch_test)
        loss += losses.item()
    loss_avg = loss/len(loader)
    
    return loss_avg


@problem.tag("hw4-A", start_line=9)
def main():
    """
    Main function of autoencoders problem.

    It should:
        A. Train an F1 model with hs 32, 64, 128, report loss of the last epoch
            and visualize reconstructions of 10 images side-by-side with original images.
        B. Same as A, but with F2 model
        C. Use models from parts A and B with h=128, and report reconstruction error (MSE) on test set.

    Note:
        - For visualizing images feel free to use images_to_visualize variable.
            It is a FloatTensor of shape (10, 784).
        - For having multiple axes on a single plot you can use plt.subplots function
        - For visualizing an image you can use plt.imshow (or ax.imshow if ax is an axis)
    """
    torch.manual_seed(20000)
    (x_train, y_train), (x_test, _) = load_dataset("mnist")
    x = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    # Neat little line that gives you one image per digit for visualization in parts a and b
    images_to_visualize = x[[np.argwhere(y_train == i)[0][0] for i in range(10)]]

    train_loader = DataLoader(TensorDataset(x), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test), batch_size=32, shuffle=True)
    
    eps = 40 
    h = (32, 64, 128)
    loss_1 = np.zeros(len(h))
    loss_2 = np.zeros(len(h))
    test_error_1 = 0
    test_error_2 = 0
    models_1 = []
    models_2 = []
    for i in range(len(h)):
        print(i)
        model_1 = F1(h[i])
        models_1.append(model_1)
        loss_1[i] = train(model_1, Adam, train_loader, eps)
        model_2 = F2(h[i])
        models_2.append(model_2)
        loss_2[i] = train(model_2, Adam, train_loader, eps) 
        if i == 2: 
            test_error_1 = evaluate(model_1, test_loader)
            test_error_2 = evaluate(model_2, test_loader)


    print("F1 training loss for h = 32 with 40 epochs:", loss_1[0])
    print("F2 training loss for h = 32 with 40 epochs:", loss_2[0])
    print("F1 training loss for h = 64 with 40 epochs:", loss_1[1])
    print("F2 training loss for h = 64 with 40 epochs:", loss_2[1])
    print("F1 training loss for h = 128 with 40 epochs:", loss_1[2])
    print("F2 training loss for h = 128 with 40 epochs:", loss_2[2])

    # h = 32
    mod1_32 = models_1[0]
    fig, axes = plt.subplots(2,10)
    fig.suptitle('F1 model with h = 32')
    for j in range(10):
        for i,ax in enumerate(axes):
            if i == 0:  
                axes[i,j].imshow(images_to_visualize[j,:].reshape(28,28))
                axes[i,j].axis('off')
            else: 
                with torch.no_grad():
                    y_hat = mod1_32(images_to_visualize[j,:])
                    axes[i,j].imshow(y_hat.reshape(28,28))
                    axes[i,j].axis('off')       
    plt.show()

    # h = 64
    mod1_64 = models_1[1]
    fig, axes = plt.subplots(2,10)
    fig.suptitle('F1 model with h = 64')
    for j in range(10):
        for i,ax in enumerate(axes):
            if i == 0:  
                axes[i,j].imshow(images_to_visualize[j,:].reshape(28,28))
                axes[i,j].axis('off')
            else: 
                with torch.no_grad():
                    y_hat = mod1_64(images_to_visualize[j,:])
                    axes[i,j].imshow(y_hat.reshape(28,28))
                    axes[i,j].axis('off')
    plt.show()

    # h = 128

    mod1_128 = models_1[2]
    fig, axes = plt.subplots(2,10)
    fig.suptitle('F1 model with h = 128')
    for j in range(10):
        for i,ax in enumerate(axes):
            if i == 0:  
                axes[i,j].imshow(images_to_visualize[j,:].reshape(28,28))
                axes[i,j].axis('off')
            else: 
                with torch.no_grad():
                    y_hat = mod1_128(images_to_visualize[j,:])
                    axes[i,j].imshow(y_hat.reshape(28,28))
                    axes[i,j].axis('off')
    plt.show()


    # h = 32
    mod2_32 = models_2[0]
    fig, axes = plt.subplots(2,10)
    fig.suptitle('F2 model with h = 32')
    for j in range(10):
        for i,ax in enumerate(axes):
            if i == 0:  
                axes[i,j].imshow(images_to_visualize[j,:].reshape(28,28))
                axes[i,j].axis('off')
            else: 
                with torch.no_grad():
                    y_hat = mod2_32(images_to_visualize[j,:])
                    axes[i,j].imshow(y_hat.reshape(28,28))
                    axes[i,j].axis('off')       
    plt.show()

    # h = 64
    mod2_64 = models_2[1]
    fig, axes = plt.subplots(2,10)
    fig.suptitle('F2 model with h = 64')
    for j in range(10):
        for i,ax in enumerate(axes):
            if i == 0:  
                axes[i,j].imshow(images_to_visualize[j,:].reshape(28,28))
                axes[i,j].axis('off')
            else: 
                with torch.no_grad():
                    y_hat = mod2_64(images_to_visualize[j,:])
                    axes[i,j].imshow(y_hat.reshape(28,28))
                    axes[i,j].axis('off')
    plt.show()

    # h = 128

    mod2_128 = models_2[2]
    fig, axes = plt.subplots(2,10)
    fig.suptitle('F2 model with h = 128')
    for j in range(10):
        for i,ax in enumerate(axes):
            if i == 0:  
                axes[i,j].imshow(images_to_visualize[j,:].reshape(28,28))
                axes[i,j].axis('off')
            else: 
                with torch.no_grad():
                    y_hat = mod2_128(images_to_visualize[j,:])
                    axes[i,j].imshow(y_hat.reshape(28,28))
                    axes[i,j].axis('off')
    plt.show()
    
    # Part c 

    print("Test error for F1 model :", test_error_1)
    print("Test error for F2 model:", test_error_2)

    
if __name__ == "__main__":
    main()
