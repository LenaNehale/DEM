import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.clipper import Clipper
from dem.models.components.noise_schedules import BaseNoiseSchedule


def wrap_for_richardsons(score_estimator):
    def _fxn(t, x, energy_function, noise_schedule, num_mc_samples):
        bigger_samples = score_estimator(t, x, energy_function, noise_schedule, num_mc_samples)

        smaller_samples = score_estimator(
            t, x, energy_function, noise_schedule, int(num_mc_samples / 2)
        )

        return (2 * bigger_samples) - smaller_samples

    return _fxn


def get_logreward_noised_samples(t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    clipper: Clipper = None,
):

    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)

    h_t = noise_schedule.h(repeated_t).unsqueeze(-1)

    samples = repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())

    log_rewards = energy_function(samples)

    if clipper is not None and clipper.should_clip_log_rewards:
        log_rewards = clipper.clip_log_rewards(log_rewards)

    return samples, log_rewards



def log_expectation_reward(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    clipper: Clipper = None,
):
    samples, log_rewards = get_logreward_noised_samples(t, x, energy_function, noise_schedule, num_mc_samples, clipper)
    return torch.logsumexp(log_rewards, dim=0)


def estimate_grad_Rt(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    is_torch_diff: bool = False,
    clipper: Clipper = None,
):
    if t.ndim == 0:
        t = t.unsqueeze(0).repeat(len(x))
    
    if is_torch_diff:      
        grad_fxn = torch.func.grad(log_expectation_reward, argnums=1)
        vmapped_fxn = torch.vmap(grad_fxn, in_dims=(0, 0, None, None, None,None), randomness="same")
        logsumexp_grad= vmapped_fxn(t, x, energy_function, noise_schedule, num_mc_samples, clipper)
        #print('grad max', logsumexp_grad.max(), 'grad min', logsumexp_grad.min(), 'grad mean', logsumexp_grad.mean(), 'grad median', logsumexp_grad.median())  
        return logsumexp_grad
        
    
    else:
        # Use the formula (9) in the idem paper 

        vmapped_fxn = torch.vmap(get_logreward_noised_samples, in_dims=(0, 0, None, None, None,None), randomness="same")
        samples, log_rewards  =  vmapped_fxn(t, x, energy_function, noise_schedule, num_mc_samples, clipper) #logrewards shape : bs, num_mc_samples
        weights = torch.softmax(log_rewards, dim=-1).unsqueeze(-1) # shape : bs, num_mc_samples, 1 
        #g = torch.func.grad(energy_function.__call__)
        #vmapped_g = torch.vmap(torch.vmap(g, in_dims=(0)), in_dims=0)
        #grad_log_rewards = vmapped_g(samples) # shape : bs, num_mc_samples, dim  
        grad_log_rewards = energy_function.score(samples) # shape : bs, num_mc_samples, dim
        SK = (grad_log_rewards*weights).sum(dim=-2)
        if torch.isnan(SK).any():
            print("NANs in score!")
            grad_log_rewards[grad_log_rewards >= 1e8] = 1e8
            grad_log_rewards[grad_log_rewards <= - 1e8] = -1e8
        if torch.isinf(SK).any():
            print("INFs in score!")
        #print('grad max', SK.max(), 'grad min', SK.min(), 'grad mean', SK.mean(), 'grad median', SK.median())  
        return SK #bs, dim  


