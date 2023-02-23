"""Utility functions for performative prediction demo."""
import numpy as np
import torch
import torch.nn as nn


def sigmoid_eps(x, epsilon):
#     return 1/(torch.exp(-x)+(1/(1-epsilon)))
    return torch.sigmoid(x)*(1-epsilon)


class onelayer_NN(nn.Module):
    
    def __init__(self, epsilon, mode):
        super().__init__()
        self.epsilon = epsilon
        if mode == 'case1':
            self.theta = nn.parameter.Parameter(data=torch.ones((1,))*torch.atanh(torch.tensor(0.5)), requires_grad=True)
        elif mode == 'RS':
#             self.theta = nn.parameter.Parameter(data=torch.ones((11,), dtype = torch.float64)*torch.atanh(torch.tensor(0.5)), requires_grad=True)
            self.theta = nn.parameter.Parameter(data = torch.normal(0.0, 1.0, (11,), dtype = torch.float64), requires_grad = True)
            
        
        self.mode = mode
        
    def forward(self, X):
        
        if self.mode == 'case1':
            print('casel')
            y_hat = X[:,0] * (torch.tanh(self.theta[0])+2)/self.epsilon
            return y_hat
        
        elif self.mode == 'RS':
#             print('in RS in forward')

            y_hat = X @ self.theta

            return sigmoid_eps(y_hat, self.epsilon)
        
#             print(f'sigmoid:{torch.sigmoid(y_hat)}')
#             return torch.maximum(torch.ones_like(y_hat) * self.epsilon, torch.sigmoid(y_hat))
 
class twolayers_NN(nn.Module):
    
    def __init__(self, epsilon, mode):
        super().__init__()
        self.epsilon = epsilon
        if mode == 'case1':
            self.theta = nn.parameter.Parameter(data=torch.ones((1,))*torch.atanh(torch.tensor(0.5)), requires_grad=True)
        elif mode == 'RS':
#             self.theta = nn.parameter.Parameter(data=torch.ones((11,), dtype = torch.float64)*torch.atanh(torch.tensor(0.5)), requires_grad=True)
            
            self.fc1 = nn.Linear(11, 8, dtype = torch.float64)
            self.leaky_relu = nn.LeakyReLU()
            self.fc2 = nn.Linear(8, 1, dtype = torch.float64)
            
        
        self.mode = mode
        
    def forward(self, X):
        if self.mode == 'case1':
            
            print('casel')
            y_hat = X[:,0] * (torch.tanh(self.theta[0])+2)/self.epsilon
            return y_hat
        
        elif self.mode == 'RS':
#             print('in RS in forward')

            fc1 = self.fc1(X)
            y_hat = torch.squeeze(self.fc2(self.leaky_relu(fc1)))
            return sigmoid_eps(y_hat, self.epsilon)
        
#             print(f'sigmoid:{torch.sigmoid(y_hat)}')
#             return torch.maximum(torch.ones_like(y_hat) * self.epsilon, torch.sigmoid(y_hat))
    
    
    

def evaluate_logistic_loss(X, Y, l2_penalty, model, eval_bool, mode='RS'):
    """Compute the l2-penalized logistic loss function

    Parameters
    ----------
        X: np.ndarray
            A [num_samples, num_features] matrix of features. The last
            feature dimension is assumed to be the bias term.
        Y: np.ndarray
            A [num_samples] vector of binary labels.
        theta: np.ndarray
            A [num_features] vector of classifier parameters
        l2_penalty: float
            Regularization coefficient. Use l2_penalty=0 for no regularization.

    Returns
    -------
        loss: float

    """
#     n = X.shape[0]
    
    if mode == 'default':
        n = X.shape
        logits = X @ theta
        log_likelihood = (
            1.0 / n * np.sum(-1.0 * np.multiply(Y, logits) + np.log(1 + np.exp(logits)))
        )

        regularization = (l2_penalty / 2.0) * np.linalg.norm(theta[:-1]) ** 2

        return log_likelihood + regularization

    elif mode == 'case1':
        if eval_bool:
            with torch.no_grad():
                y_hat = model(X)
        else:
            y_hat = model(X)
        n = y_hat.shape[0]
#         print('in evaluate logistic loss and case1')
        loss = -15/4 * (y_hat - Y) + 0.5 * (y_hat**2) + 0.5 * (Y**2) + (15/4)**2
        return 1.0/n * torch.sum(loss)
    
    elif mode =='RS':
        if eval_bool:
            with torch.no_grad():
                y_hat = model(X)
        else:
            y_hat = model(X)
        n = y_hat.shape[0]
        loss = 0.5 * (y_hat - Y )**2
        return torch.mean(loss)
#         return 1.0/n * torch.sum(loss)

      
    
#     '''


def fit_logistic_regression(X, Y, l2_penalty, model=None, optimizer=None, eval_bool=False, epsilon=0, tol=1e-9, theta_init=None, mode = 'RS'):
    """Fit a logistic regression model via gradient descent.

    Parameters
    ----------
        X: np.ndarray
            A [num_samples, num_features] matrix of features.
            The last feature dimension is assumed to be the bias term.
        Y: np.ndarray
            A [num_samples] vector of binary labels.
        l2_penalty: float
            Regularization coefficient. Use l2_penalty=0 for no regularization.
        tol: float
            Stopping criteria for gradient descent
        theta_init: np.ndarray
            A [num_features] vector of classifier parameters to use a
            initialization

    Returns
    -------
        theta: np.ndarray
            The optimal [num_features] vector of classifier parameters.

    """
#     X = np.copy(X)
#     Y = np.copy(Y)
    n, d = X.shape
    
    if mode == 'default':
        

        # Smoothness of the logistic loss
        smoothness = np.sum(X ** 2) / (4.0 * n)

        # Optimal initial learning rate
        eta_init = 1 / (smoothness + l2_penalty)

        if theta_init is not None:
            theta = np.copy(theta_init)
        else:
            theta = np.zeros(d)

        # Evaluate loss at initialization
        prev_loss = evaluate_logistic_loss(X, Y, theta, l2_penalty)

        loss_list = [prev_loss]
        i = 0
        gap = 1e30

        eta = eta_init
        while gap > tol:

            # take gradients
            exp_tx = np.exp(X @ theta)
            c = exp_tx / (1 + exp_tx) - Y
            gradient = 1.0 / n * np.sum(
                X * c[:, np.newaxis], axis=0
            ) + l2_penalty * np.append(theta[:-1], 0)

            new_theta = theta - eta * gradient

            # compute new loss
            loss = evaluate_logistic_loss(X, Y, new_theta, l2_penalty)

            # do backtracking line search
            if loss > prev_loss:
                eta = eta * 0.1
                gap = 1e30
                continue

            eta = eta_init
            theta = np.copy(new_theta)

            loss_list.append(loss)
            gap = prev_loss - loss
            prev_loss = loss

            i += 1

        return theta
    
    elif mode == 'case1': # Considering x and theta to be scalar
        
#         print('in fit and case1')

#         eta_init = 1e-2

#         if theta_init is not None:
#             theta = theta_init.clone()
#         else:
#             theta = torch.zeros((1,))
#             theta[0] = torch.atanh(0.5)

#         theta = theta_init.clone()
        # Evaluate loss at initialization
#         y_hat = model(X)
        prev_loss = evaluate_logistic_loss(X, Y, l2_penalty, model, eval_bool, mode = mode)

        loss_list = [prev_loss.item()]
        i = 0
        gap = 1e30

#         eta = eta_init

    #     print('in fit') 

        while gap > tol:
            
            # compute gradient
            optimizer.zero_grad()
#             theta_0 = model.theta[0]
            prev_loss.backward()
            optimizer.step()
        
            # compute new loss
            loss = evaluate_logistic_loss(X, Y, l2_penalty, model, eval_bool, mode = mode)

            loss_list.append(loss.item())
            gap = np.abs(prev_loss.item() - loss.item())
            prev_loss = loss

            i += 1
            
        print(i)
 
        return model.theta.detach().numpy()
     

    elif mode == 'RS': 
        
        # Evaluate loss at initialization
#         y_hat = model(X)
#         print(labels)
#         print(f'Y type:{Y.dtype}')

        prev_loss = evaluate_logistic_loss(X, Y, l2_penalty, model, eval_bool, mode = mode)
        loss_list = [prev_loss.item()]
        i = 0
        gap = 1e30


        while gap > tol:
            
            # compute gradient
            optimizer.zero_grad()
#             theta_0 = model.theta[0]
            prev_loss.backward()
            optimizer.step()
#             if i < 10:
#                 with torch.no_grad():
#                     print(model.theta.grad)
        
            # compute new loss
            loss = evaluate_logistic_loss(X, Y, l2_penalty, model, eval_bool, mode = mode)

            loss_list.append(loss.item())
#             print(loss.item())
            gap = np.abs(prev_loss.item() - loss.item())
            prev_loss = loss

            i += 1
#             print(model.theta.detach().numpy())
            
#         print(i)
        
#         preds = model(X)
#         print(f'Y type:{Y.dtype}, pred type:{preds.dtype}')
#         print(((preds > (1.0-model.epsilon)/2.0)*(1.0-model.epsilon)).dtype)
#         y_pred = ((preds > (1.0-model.epsilon)/2.0)*(1.0-model.epsilon)).float()
#         print(y_pred.dtype)
#         res = y_pred == Y.float()
#         acc = torch.mean(res*1.0).item()
#         print(f'accuracy in fit_loss:{acc}')
#         return model.theta.detach().numpy()

