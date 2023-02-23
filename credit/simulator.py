"""Implementation of the Perdomo et. al model of strategic classification.

The data is from the Kaggle Give Me Some Credit dataset:

    https://www.kaggle.com/c/GiveMeSomeCredit/data,

and the dynamics are taken from:

    Perdomo, Juan C., Tijana Zrnic, Celestine Mendler-Dünner, and Moritz Hardt.
    "Performative Prediction." arXiv preprint arXiv:2002.06673 (2020).
"""
import copy
import dataclasses
from typing import Any

import whynot as wn
import whynot.traceable_numpy as np
from whynot.dynamics import BaseConfig, BaseIntervention, BaseState
from whynot.simulators.credit.dataloader import CreditData
from examples.dynamic_decisions.scripts.utils_torch import Model
import torch.nn as nn
import torch

def sigmoid_eps(x, epsilon):
    return 1/(np.exp(-x)+(1/(1-epsilon)))

def forward(X, theta, epsilon):
    return sigmoid_eps(X @ theta, epsilon)

@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """State of the Credit model."""

    #: Matrix of agent features (e.g. https://www.kaggle.com/c/GiveMeSomeCredit/data)
    features: np.ndarray = CreditData.features

    #: Vector indicating whether or not the agent experiences financial distress
    labels: np.ndarray = CreditData.labels

    def values(self):
        """Return the state as a dictionary of numpy arrays."""
        return {name: getattr(self, name) for name in self.variable_names()}


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of Credit simulator dynamics.

    Examples
    --------
    >>> # Configure simulator for run for 10 iterations
    >>> config = Config(start_time=0, end_time=10, delta_t=1)

    """

    # Dynamics parameters
    #: Subset of the features that can be manipulated by the agent
    changeable_features: np.ndarray = np.array([0, 5, 7])

    #: Model how much the agent adapt her features in response to a classifier
    epsilon: float = 0.1

    #: Parameters for logistic regression classifier used by the institution
    theta: np.ndarray = np.ones((11,)) # to change this, also change whynot/simulators/credit/environments.py --> credit_action_space

    #: L2 penalty on the logistic regression loss
    l2_penalty: float = 0.0

    #: Whether or not dynamics have memory
    memory: bool = False

    #: State systems resets to if no memory.
    base_state: Any = State()

    # Simulator book-keeping
    #: Start time of the simulator
    start_time: int = 0
    #: End time of the simulator
    end_time: int = 5
    #: Spacing of the evaluation grid
    delta_t: int = 1
        
    mode: str = 'default'
        
#     model: nn.Module = Model(epsilon, mode)


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the Credit model.

    An intervention changes a subset of the configuration variables in the
    specified year. The remaining variables are unchanged.

    Examples
    --------
    >>> # Starting at time 25, update the classifier to random chance.
    >>> config = Config()
    >>> Intervention(time=25, theta=np.zeros_like(config.theta))

    """

    def __init__(self, time=30, **kwargs):
        """Specify an intervention in credit.

        Parameters
        ----------
            time: int
                Time of intervention in simulator dynamics.
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


def logistic_loss(config, features, labels, theta):
    """Evaluate the performative loss for logistic regression classifier."""
    
    config = config.update(Intervention(theta=theta))
    
    X = np.copy(features)
    Y = np.copy(labels)
    n = X.shape[0]
    
    if config.mode == 'default':

        logits = X @ config.theta
        log_likelihood = (
            1.0 / n * np.sum(-1.0 * np.multiply(Y, logits) + np.log(1 + np.exp(logits)))
        )

        regularization = (config.l2_penalty / 2.0) * np.linalg.norm(config.theta[:-1]) ** 2

        return log_likelihood + regularization

    elif config.mode == 'case1':
        
#         print('in logistic loss and case1')
        
        theta_0 = theta[0]
        y_hat = X[:,0] * (np.tanh(theta_0)+2)/config.epsilon
        loss = -15/4 * (y_hat - Y) + 1/2 * (y_hat**2) + 1/2 * (Y**2) + (15/4)**2
        return 1.0/n * np.sum(loss)
    
    elif config.mode =='RS':
        
#         print(f'mode:{config.mode}')
        y_hat = forward(X, config.theta, config.epsilon)
#         y_hat = sigmoid_eps(X @ config.theta, config.epsilon)
        n = y_hat.shape[0]
#         print('in evaluate logistic loss and case1')
        loss = 0.5 * (y_hat - Y)**2
#         loss = -0.5 * (y_hat - Y) + 0.5 * (y_hat**2) + 0.5 * (Y**2) + (0.5)**2

#         loss = -1.0 * np.multiply(Y, y_hat) + np.log(1 + np.exp(y_hat)) + ((config.l2_penalty / 2.0) * (y_hat - Y)**2)
        return 1.0/n * np.sum(loss)

    

def agent_model(features, config):
    """Compute agent reponse to the classifier and adapt features accordingly.

    TODO: For now, the best-response model corresponds to best-response with
    linear utility and quadratic costs. We should expand this to cover a rich
    set of agent models beyond linear/quadratic, and potentially beyond
    best-response.
    """
    
    if config.mode == 'default':
        
        # Move everything by epsilon in the direction towards better classification
        strategic_features = np.copy(features)
        theta_strat = config.theta[config.changeable_features].flatten()
#         print(f'config.changeable_features:{config.changeable_features}, theta_strat:{theta_strat.shape}')
        strategic_features[:, config.changeable_features] -= config.epsilon * theta_strat
        return strategic_features
#         return np.ones_like(strategic_features, dtype=np.float32)
    
    elif config.mode == 'case1': # W1 counterexample
#         print('in agent_model and case1')
        strategic_features = np.copy(features)    
        strategic_features[:,0] = (np.tanh(config.theta[0])+2) * config.epsilon
        return strategic_features
    
    elif config.mode == 'RS': # Rejection Sampling case
#         print(f'features:{features[:50]}')
#         print(f'config theta:{config.theta}')

#         preds = sigmoid_eps(features @ config.theta, config.epsilon)
#         print(f'preds:{preds}')
#         strategic_features = np.copy(features)
# #         f_theta_strat = preds[config.changeable_features].flatten()
# #         print(f'config.changeable_features:{config.changeable_features}, theta_strat:{theta_strat.shape}')
#         strategic_features[:, config.changeable_features] -= config.epsilon * preds[:, np.newaxis]
#         return strategic_features
    
    
        
        preds = forward(features, config.theta, config.epsilon)
        print(f'preds:{preds}')
        n = features.shape[0]
        r = np.random.uniform(0,1,(n,))
        resample_indices = np.random.randint(0,n,(n,))
        new_indices = np.where(r < (preds+config.epsilon+1e-3), resample_indices, np.arange(n))
#         new_features = features[new_indices]

        
#         change strategic features 
        strategic_features = features[:,config.changeable_features]
        new_strategic_features = strategic_features[new_indices]
        new_features = np.copy(features)
        new_features[:, config.changeable_features] = new_strategic_features
#         print(f'new_indices:{new_indices}')

        
        # change non-strategic features
#         d = features.shape[1]
#         all_indices = np.arange(d)
#         non_str_indices = np.array([i for i in all_indices if i not in config.changeable_features])
#         non_str_features = features[:, non_str_indices]
#         new_non_str_features = non_str_features[new_indices]
#         new_features = np.copy(features)
#         new_features[:, non_str_indices] = new_non_str_features
#         print(f'new_indices:{new_indices}')
        
        return new_features
        
    


def dynamics(state, time, config, intervention=None):
    """Perform one round of interaction between the agents and the credit scorer.

    Parameters
    ----------
        state: whynot.simulators.credit.State
            Agent state at time TIME
        time: int
            Current round of interaction
        config: whynot.simulators.credit.Config
            Configuration object controlling the interaction, e.g. classifier
            and agent model
        intervention: whynot.simulators.credit.Intervention
            Intervention object specifying when and how to update the dynamics.

    Returns
    -------
        state: whynot.simulators.credit.State
            Agent state after one step of strategic interaction.

    """
    if intervention and time >= intervention.time:
        config = config.update(intervention)

    # Only use the current state if the dynamics have memory.
    # Otherwise, agents "reset" to the base dataset. The latter
    # case is the one treated in the performative prediction paper.
    if config.memory:
        features, labels = state.features, state.labels
    else:
        features, labels = config.base_state.features, config.base_state.labels

    # Update features in response to classifier. Labels are fixed.
    strategic_features = agent_model(features, config)
    return strategic_features, labels


def simulate(initial_state, config, intervention=None, seed=None):
    """Simulate a run of the Credit model.

    Parameters
    ----------
        initial_state: whynot.credit.State
        config: whynot.credit.Config
            Base parameters for the simulator run
        intervention: whynot.credit.Intervention
            (Optional) Parameters specifying a change in dynamics
        seed: int
            Unused since the simulator is deterministic.

    Returns
    -------
        run: whynot.dynamics.Run
            Simulator rollout

    """
    # Iterate the discrete dynamics
    times = [config.start_time]
    states = [initial_state]
    state = copy.deepcopy(initial_state)

    for step in range(config.start_time, config.end_time):
        next_state = dynamics(state, step, config, intervention)
        state = State(*next_state)
        states.append(state)
        times.append(step + 1)

    return wn.dynamics.Run(states=states, times=times)


if __name__ == "__main__":
    print(simulate(State(), Config(end_time=2)))
