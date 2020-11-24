from typing import NamedTuple, Tuple

import torch
import torch.jit

from norse.torch.functional.threshold import threshold


class LIFExParameters(NamedTuple):
    """Parametrization of an Exponential Leaky Integrate and Fire neuron

    Parameters:
        delta_T (torch.Tensor): sharpness or speed of the exponential growth in mV
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        v_leak (torch.Tensor): leak potential in mV
        v_th (torch.Tensor): threshold potential in mV
        v_reset (torch.Tensor): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (float): hyper parameter to use in surrogate gradient computation
    """

    delta_T: torch.Tensor = torch.as_tensor(0.5)
    tau_syn_inv: torch.Tensor = torch.as_tensor(1.0 / 5e-3)
    tau_mem_inv: torch.Tensor = torch.as_tensor(1.0 / 1e-2)
    v_leak: torch.Tensor = torch.as_tensor(0.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    v_reset: torch.Tensor = torch.as_tensor(0.0)
    method: str = "super"
    alpha: float = 0.0


class LIFExState(NamedTuple):
    """State of a LIFEx neuron

    Parameters:
        z (torch.Tensor): recurrent spikes
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
    """

    z: torch.Tensor
    v: torch.Tensor
    i: torch.Tensor


class LIFExFeedForwardState(NamedTuple):
    """State of a feed forward LIFEx neuron

    Parameters:
        v (torch.Tensor): membrane potential
        i (torch.Tensor): synaptic input current
    """

    v: torch.Tensor
    i: torch.Tensor


def lif_ex_step(
    input_tensor: torch.Tensor,
    state: LIFExState,
    input_weights: torch.Tensor,
    recurrent_weights: torch.Tensor,
    p: LIFExParameters = LIFExParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFExState]:
    r"""Computes a single euler-integration step of an exponential LIF neuron-model
    adapted from https://neuronaldynamics.epfl.ch/online/Ch5.S2.html. More
    specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right)\right) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + w_{\text{input}} z_{\text{in}} \\
            i &= i + w_{\text{rec}} z_{\text{rec}}
        \end{align*}

    where :math:`z_{\text{rec}}` and :math:`z_{\text{in}}` are the recurrent
    and input spikes respectively.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        s (LIFExState): current state of the LIF neuron
        input_weights (torch.Tensor): synaptic weights for incoming spikes
        recurrent_weights (torch.Tensor): synaptic weights for recurrent spikes
        p (LIFExParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    # compute voltage updates
    dv_leak = p.v_leak - state.v
    dv_exp = p.delta_T * torch.exp((state.v - p.v_th) / p.delta_T)
    dv = dt * p.tau_mem_inv * (dv_leak + dv_exp + state.i)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute current jumps
    i_new = (
        i_decayed
        + torch.nn.functional.linear(input_tensor, input_weights)
        + torch.nn.functional.linear(state.z, recurrent_weights)
    )

    return z_new, LIFExState(z_new, v_new, i_new)


def lif_ex_feed_forward_step(
    input_tensor: torch.Tensor,
    state: LIFExFeedForwardState = LIFExFeedForwardState(0, 0),
    p: LIFExParameters = LIFExParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, LIFExFeedForwardState]:
    r"""Computes a single euler-integration step of an exponential LIF neuron-model
    adapted from https://neuronaldynamics.epfl.ch/online/Ch5.S2.html.
    It takes as input the input current as generated by an arbitrary torch
    module or function. More specifically it implements one integration step of
    the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right)\right) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    together with the jump condition

    .. math::
        z = \Theta(v - v_{\text{th}})

    and transition equations

    .. math::
        \begin{align*}
            v &= (1-z) v + z v_{\text{reset}} \\
            i &= i + i_{\text{in}}
        \end{align*}

    where :math:`i_{\text{in}}` is meant to be the result of applying an
    arbitrary pytorch module (such as a convolution) to input spikes.

    Parameters:
        input_tensor (torch.Tensor): the input spikes at the current time step
        state (LIFExFeedForwardState): current state of the LIF neuron
        p (LIFExParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    # compute voltage updates
    dv_leak = p.v_leak - state.v
    dv_exp = p.delta_T * torch.exp((state.v - p.v_th) / p.delta_T)
    dv = dt * p.tau_mem_inv * (dv_leak + dv_exp + state.i)
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute new spikes
    z_new = threshold(v_decayed - p.v_th, p.method, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute current jumps
    i_new = i_decayed + input_tensor

    return z_new, LIFExFeedForwardState(v_new, i_new)


def lif_ex_current_encoder(
    input_current: torch.Tensor,
    voltage: torch.Tensor,
    p: LIFExParameters = LIFExParameters(),
    dt: float = 0.001,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes a single euler-integration step of a leaky integrator
    adapted from https://neuronaldynamics.epfl.ch/online/Ch5.S2.html. More
    specifically it implements one integration step of the following ODE

    .. math::
        \begin{align*}
            \dot{v} &= 1/\tau_{\text{mem}} \left(v_{\text{leak}} - v + i + \Delta_T exp\left({{v - v_{\text{th}}} \over {\Delta_T}}\right)\right) \\
            \dot{i} &= -1/\tau_{\text{syn}} i
        \end{align*}

    Parameters:
        input (torch.Tensor): the input current at the current time step
        voltage (torch.Tensor): current state of the LIFEx neuron
        p (LIFExParameters): parameters of a leaky integrate and fire neuron
        dt (float): Integration timestep to use
    """
    dv_leak = p.v_leak - voltage
    dv_exp = p.delta_T * torch.exp((voltage - p.v_th) / p.delta_T)
    dv = dt * p.tau_mem_inv * (dv_leak + dv_exp + input_current)
    voltage = voltage + dv
    z = threshold(voltage - p.v_th, p.method, p.alpha)

    voltage = voltage - z * (voltage - p.v_reset)
    return z, voltage