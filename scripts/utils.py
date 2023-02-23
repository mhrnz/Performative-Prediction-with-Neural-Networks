"""Utility functions for performative prediction demo."""
import numpy as np


def evaluate_logistic_loss(X, Y, theta, l2_penalty, epsilon=0, mode='default'):
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
    n = X.shape[0]
    
    if mode == 'default':
        logits = X @ theta
        log_likelihood = (
            1.0 / n * np.sum(-1.0 * np.multiply(Y, logits) + np.log(1 + np.exp(logits)))
        )

        regularization = (l2_penalty / 2.0) * np.linalg.norm(theta[:-1]) ** 2

        return log_likelihood + regularization

    elif mode == 'case1':
        
#         print('in evaluate logistic loss and case1')
        theta_0 = theta[0]
        y_hat = X[:,0] * (np.tanh(theta_0)+2)/epsilon
        loss = -15/4 * (y_hat - Y) + 0.5 * (y_hat**2) + 0.5 * (Y**2) + (15/4)**2
        return 1.0/n * np.sum(loss)

      
    
#     '''


def fit_logistic_regression(X, Y, l2_penalty, epsilon=0, tol=1e-9, theta_init=None, mode = 'default'):
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
    X = np.copy(X)
    Y = np.copy(Y)
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

        eta_init = 1e-2

        if theta_init is not None:
            theta = np.copy(theta_init)
        else:
            theta = np.zeros((1,))
#             theta[0] = np.arctanh(0.5)
            theta[0] = np.arctanh(0.5)

        # Evaluate loss at initialization
        prev_loss = evaluate_logistic_loss(X, Y, theta, l2_penalty, epsilon, mode = mode)

        loss_list = [prev_loss]
        i = 0
        gap = 1e30

        eta = eta_init

    #     print('in fit') 

        while gap > tol:
            
#             print(f'theta:{theta}')
            theta_0 = theta[0]
            y_hat = X[:,0] * (np.tanh(theta_0)+2)/epsilon
            a = X[:,0]/epsilon * (1-np.tanh(theta_0)**2)
#             print(f'in fit, theta_0:{theta_0}, a:{a}')
            grad = np.sum((-15/4 + y_hat) * a) * 1.0/n
#             print(f'grad:{grad}')
            gradient = np.zeros((1,))
            gradient[0] = grad
            
            new_theta = theta - eta * gradient
#             print(f'in fit, theta:{theta}, new_theta:{new_theta}')
#             print(f'new_theta:{new_theta}')

            # compute new loss
            loss = evaluate_logistic_loss(X, Y, new_theta, l2_penalty, epsilon, mode = mode)

            # do backtracking line search
#             if loss > prev_loss:
#                 eta = eta * 0.9
#                 gap = 1e30
#                 continue

#             eta = eta_init
            theta = np.copy(new_theta)

            loss_list.append(loss)
            gap = np.abs(prev_loss - loss)
            prev_loss = loss

            i += 1

        return theta
    
#     elif mode == 'case2':

#     X = np.copy(X)
#     Y = np.copy(Y)
#     n, d = X.shape

#     eta_init = 1e-1

#     if theta_init is not None:
#         theta = np.copy(theta_init)
#     else:
#         theta = np.zeros(d)
#         theta[0] = np.arctanh(0.25)

#     # Evaluate loss at initialization
#     prev_loss = evaluate_logistic_loss(X, Y, theta, l2_penalty, epsilon)

#     loss_list = [prev_loss]
#     i = 0
#     gap = 1e30

#     eta = eta_init
    
# #     print('in fit') 
    
#     while gap > tol:
        
        
# #         ''' Old gradient 
        
# #         # take gradients
# #         exp_tx = np.exp(X @ theta)
# #         c = exp_tx / (1 + exp_tx) - Y
# #         gradient = 1.0 / n * np.sum(
# #             X * c[:, np.newaxis], axis=0
# #         )


# # #         # take gradients
# # #         exp_tx = np.exp(X @ theta)
# # #         c = exp_tx / (1 + exp_tx) - Y
# # #         gradient = 1.0 / n * np.sum((1-np.tanh(theta)**2)[np.newaxis,:] *
# # #             X * c[:, np.newaxis], axis=0
# # #         ) + l2_penalty * np.append(theta[:-1], 0) * np.append((1-np.tanh(theta[:-1])**2),0)
# #         '''
        
# # #         ''' New gradient
        
# # #         a = X @ np.diag(1- (np.tanh(theta)**2)) / epsilon
# # #         b = -9/16 + X @ (np.tanh(theta)+2)/epsilon 
# # #         gradient = 1.0/n * np.sum(a * b[:, np.newaxis], axis=0)
        
# # #         '''
#         theta_0 = theta[0]
#         y_hat = X[:,0] * (np.tanh(theta_0)+2)/epsilon
#         a = X[:,0]/epsilon * (1-np.tanh(theta_0)**2)
#         grad = np.sum((-15/4 + y_hat) * a) * 1.0/n
        
#         gradient = np.zeros(d)
#         gradient[0] = grad

# #         print(f'gradient:{gradient}')

#         new_theta = theta - eta * gradient


#         # compute new loss
#         loss = evaluate_logistic_loss(X, Y, new_theta, l2_penalty, epsilon)

#         # do backtracking line search
#         if loss > prev_loss:
#             eta = eta * 0.1
#             gap = 1e30
#             continue

#         eta = eta_init
#         theta = np.copy(new_theta)

#         loss_list.append(loss)
#         gap = prev_loss - loss
#         prev_loss = loss

#         i += 1

#     return theta
