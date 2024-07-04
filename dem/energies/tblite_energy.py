from tblite.interface import Structure, Calculator
import pickle
import torch
from wurlitzer import pipes 
import numpy as np
import argparse
import os
from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.data_utils import remove_mean
import PIL
import matplotlib.pyplot as plt
from lightning.pytorch.loggers import WandbLogger


class TBLiteEnergy(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality,
        n_particles,
        data_path,
        data_path_train=None,
        data_path_val=None,
        device="cpu",
        plot_samples_epoch_period=5,
        plotting_buffer_sample_size=512,
        data_normalization_factor=1.0,
        is_molecule=True, 
        T = 0.1
    ):

        self._dimensionality = dimensionality
        self.n_particles = n_particles
        self.n_dims = self.dimensionality//self.n_particles
        self.n_spatial_dim = self.n_dims
        self.data_path_train = data_path_train
        self.data_path_val = data_path_val
        self.data_path = data_path
        self.T = T
        self.device=device
        self.atom_ids = pickle.load(open('/home/mila/l/lena-nehale.ezzine/Amgen/DEM/data/atom_ids.pkl', "rb"))

        self.curr_epoch = 0
        self.plot_samples_epoch_period = 5
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period
        self.data_normalization_factor = data_normalization_factor
        self.xtb_accuracy = 1e4
        self.calculator='GFN2-xTB'
        
        super().__init__(dimensionality=dimensionality, is_molecule=is_molecule)

    def get_batch_shape(self, x: torch.Tensor) -> torch.Size:
        if x.shape[-1] == self.n_dims:
            batch_shape = x.shape[: -2]
        elif x.shape[-1] == self.n_particles * self.n_dims:
            batch_shape = x.shape[: -1]
        if batch_shape == torch.Size([]):
            batch_shape = (1,)
        return batch_shape
    
    def get_logrew(self,coords):
        with pipes():
            try:
                calc = Calculator(self.calculator, self.atom_ids, coords.detach().cpu().numpy()* 1.8897259886)
                calc.set('accuracy', self.xtb_accuracy) # Default is 1
                res = calc.singlepoint()
                logrew = - res.get("energy")/self.T
            except RuntimeError:
                logrew = - 1e5/self.T
        print('logrew', logrew)
        return  logrew





    def get_gradient(self,coords):
        with pipes():
            try:
                calc = Calculator(self.calculator, self.atom_ids, coords.detach().cpu().numpy()* 1.8897259886)
                calc.set('accuracy', self.xtb_accuracy ) # Default is 1
                res = calc.singlepoint()
                grad =  - torch.Tensor(res.get("gradient")/self.T).flatten()
            except RuntimeError:
                # Choose random gradient to perturb the configuration with non-converging energy.
                new_coords = torch.rand(3*len(self.atom_ids))*10
                grad = self.get_gradient(new_coords)
            print('grad')
        return grad



        

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = self.get_batch_shape(x)
        xx = x.view(-1, self.n_particles, self.n_dims)
        logrews = torch.Tensor([self.get_logrew(sample) for sample in xx])
        logrews = logrews.view(*batch_shape).to(self.device)
        if batch_shape == (1,):
            return logrews[0]
        else:
            return logrews
        
    
    
    def score(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = self.get_batch_shape(x)
        xx = x.view(-1, self.n_particles, self.n_dims)
        grads = torch.stack([self.get_gradient(sample) for sample in xx])
        if x.shape[-1] == self.n_dims:
            grads = grads.view(*batch_shape, self.n_particles, self.n_dims)
        elif x.shape[-1] == self.n_particles * self.n_dims:
            grads = grads.view(*batch_shape, self.n_particles*self.n_dims)
        grads = grads.to(self.device)
        if batch_shape == (1,):
            return grads[0]
        else:
            return grads
    
    def setup_test_set(self):
        if self.data_path_val is None:
            raise ValueError("Data path for validation data is not provided")
        data = np.load(self.data_path_val, allow_pickle=True)
        data = remove_mean(data, self.n_particles, self.n_dims)
        data = torch.tensor(data, device=self.device)
        return data

    def setup_val_set(self):
        if self.data_path_val is None:
            raise ValueError("Data path for validation data is not provided")
        data = np.load(self.data_path_val, allow_pickle=True)
        data = remove_mean(data, self.n_particles, self.n_dims)
        data = torch.tensor(data, device=self.device)
        return data
    




    def interatomic_dist(self, x):
        batch_shape = self.get_batch_shape(x)
        x = x.view(*batch_shape, self.n_particles, self.n_dims)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1) == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if latest_samples is None:
            return

        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            samples_fig = self.get_dataset_fig(latest_samples)

            wandb_logger.log_image(f"{prefix}generated_samples", [samples_fig])

            if cfm_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(cfm_samples)

                wandb_logger.log_image(f"{prefix}cfm_generated_samples", [cfm_samples_fig])

        self.curr_epoch += 1

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        samples_fig = self.get_dataset_fig(samples)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def get_dataset_fig(self, samples):
        test_data_smaller = self.sample_test_set(10)  # TODO : change size of sampled test set

        fig, axs = plt.subplots(1, 2, figsize=(12, 4)) 

        dist_samples = self.interatomic_dist(samples).detach().cpu()
        dist_test = self.interatomic_dist(test_data_smaller).detach().cpu()


        axs[0].hist(
            dist_samples.view(-1),
            bins=50,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].hist(
            dist_test.view(-1),
            bins=50,
            alpha=0.5,
            density=True,
            histtype="step",
            linewidth=4,
        )
        axs[0].set_xlabel("Interatomic distance")
        axs[0].legend(["generated data", "test data"])

        energy_samples = -self(samples).detach().detach().cpu()
        energy_test = -self(test_data_smaller).detach().detach().cpu()


        axs[1].hist(
            energy_test.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            color="g",
            histtype="step",
            linewidth=4,
            label="test data",
        )
        axs[1].hist(
            energy_samples.cpu(),
            bins=100,
            density=True,
            alpha=0.4,
            color="r",
            histtype="step",
            linewidth=4,
            label="generated data",
        )
        axs[1].set_xlabel("Energy")
        axs[1].legend()

        fig.canvas.draw()
        return PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

