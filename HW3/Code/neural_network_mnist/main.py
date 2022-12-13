# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.alpha0 = 1/math.sqrt(d)
        self.alpha1 = 1/math.sqrt(h)
        rand0 = Uniform(torch.tensor([-self.alpha0]), torch.tensor([self.alpha0]))
        rand1 = Uniform(torch.tensor([-self.alpha1]), torch.tensor([self.alpha1]))
        self.weight0 = rand0.sample([h,d]).view(h,d)
        self.bias0 = rand0.sample([h,])
        self.weight1 = rand1.sample([k,h]).view(k,h)
        self.bias1 = rand1.sample([k,])
        self.params = [self.weight0, self.bias0, self.weight1, self.bias1]
        for params in self.params: 
             params.requires_grad =True
             params = Parameter(params)
            
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        
        x = torch.add(torch.matmul(self.weight0,x.transpose(0,1)), self.bias0)
        x = relu(x)
        x = torch.add(torch.matmul(self.weight1,x), self.bias1)
        x = x.transpose(0,1)
        return x


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.alpha0 = 1/math.sqrt(d)
        self.alpha1 = 1/math.sqrt(h0)
        self.alpha2 = 1/math.sqrt(h1)
        rand0 = Uniform(torch.tensor([-self.alpha0]), torch.tensor([self.alpha0]))
        rand1 = Uniform(torch.tensor([-self.alpha1]), torch.tensor([self.alpha1]))
        rand2 = Uniform(torch.tensor([-self.alpha2]), torch.tensor([self.alpha2]))
        self.weight0 = rand0.sample([h0,d]).view(h0,d)
        self.bias0 = rand0.sample([h0,])
        self.weight1 = rand1.sample([h1,h0]).view(h1,h0)
        self.bias1 = rand1.sample([h1,])
        self.weight2 = rand2.sample([k,h1]).view(k,h1)
        self.bias2 = rand2.sample([k,])
        self.params = [self.weight0, self.bias0, self.weight1, self.bias1,self.weight2, self.bias2]
        for params in self.params: 
            params.requires_grad = True
            params = Parameter(params)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        x = torch.add(torch.matmul(self.weight0,x.transpose(0,1)), self.bias0)
        x = relu(x)
        x = torch.add(torch.matmul(self.weight1,x), self.bias1)
        x = relu(x)
        x = torch.add(torch.matmul(self.weight2,x), self.bias2)
        x = x.transpose(0,1)
        return x


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    
    ce_loss = []
    epochs = 100
    lr = 1e-3
    optimize= optimizer(model.params, lr)
    

    for i in range(epochs): 
        print(i)
        loss_epoch = 0
        accuracy = 0
        for batch in train_loader: 
            images, labels = batch
            images, labels = images, labels
            optimize.zero_grad()
            y_hat = model(images)
            y_pred = torch.argmax(y_hat, 1)
            accuracy +=torch.sum(y_pred == labels).item()
            loss = cross_entropy(y_hat, labels)     
            loss_epoch += loss.item()
            loss.backward()
            optimize.step()
        ce_loss.append(loss_epoch/len(train_loader))
        acc_rate = accuracy/len(train_loader.dataset)
        print(acc_rate)
        if acc_rate>0.99:
            break
    param_1=  sum(p.numel() for p in model.params)
    print("Number of parameters in model:", param_1)
    return ce_loss



@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    torch.manual_seed(20000)
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(x,y), batch_size = 32, shuffle = True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size =32, shuffle =True)
    
    d =784
    h = 64
    h0 = 32
    h1 = 32
    k = 10
    f1 = F1(h,d,k)
    f2 = F2(h0,h1,d,k)

    err1 = train(f1, Adam, train_loader)
    err2 = train(f2,Adam, train_loader)

    x_1 = range(len(err1))
    x_2 = range(len(err2))

    losses_1 = 0
    losses_2 = 0
    test_acc_F1 = 0
    test_acc_F2 = 0
    for batch_test in test_loader:
        images_test, labels_test = batch_test
        images_test, labels_test = images_test, labels_test
        y_hat1 = f1(images_test)
        y_hat2 = f2(images_test)
        y_pred1 = torch.argmax(y_hat1,1)
        y_pred2 = torch.argmax(y_hat2, 1)
        test_acc_F1 +=torch.sum(y_pred1 == labels_test).item()
        test_acc_F2 +=torch.sum(y_pred2 == labels_test).item() 
        loss_1 = cross_entropy(y_hat1, labels_test)
        loss_2 = cross_entropy(y_hat2, labels_test)
        losses_1 += loss_1.item()
        losses_2 += loss_2.item()
    loss1 = loss_1/len(test_loader)
    loss2 = loss_2/len(test_loader)
    acc_rate_1 = test_acc_F1/len(test_loader.dataset)
    acc_rate_2 = test_acc_F2/len(test_loader.dataset)

    print("Training loss of F1 model:", err1[-1])
    print("Training loss of F2 model:", err2[-1])
    print("Test loss of F1 model:", loss1)
    print("Test loss of F2 model:", loss2)
    print("Test accuracy of F1 model:", acc_rate_1)
    print("Test accuracy of F2 model:", acc_rate_2)


   
    
    plt.figure()
    plt.plot(x_1, err1, 'rx-')
    plt.title('F1 Model')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.show()

    plt.figure()
    plt.plot(x_2, err2,'rx-')
    plt.title('F2 Model')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.show()


if __name__ == "__main__":
    main()
