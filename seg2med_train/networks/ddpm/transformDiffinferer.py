from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import Inferer
from generative.inferers.inferer import DiffusionInferer
from monai.utils import optional_import

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

class transformDiffusionInferer(DiffusionInferer):
    """
    DiffusionInferer takes a trained diffusion model and a scheduler and can be used to perform a signal forward pass
    for a training iteration, and sample from the model.


    Args:
        scheduler: diffusion scheduler.
    """

    def __init__(self, scheduler: nn.Module) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    def __call__(
        self,
        inputs: torch.Tensor,
        append_image: torch.tensor,
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added.
            append_image: the original image that should be transformed
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the input.
            timesteps: random timesteps.
            condition: Conditioning for network input.
        """
        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
        noisy_image = torch.cat((noisy_image,append_image),1)
        #print("noisy image shape after concatenating:", noisy_image.shape)
        prediction = diffusion_model(x=noisy_image, timesteps=timesteps, context=condition)

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        input_image: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            input_image: input image that should be to target transformed.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        if not scheduler:
            scheduler = self.scheduler
        image = torch.cat((input_noise,input_image),1)
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            model_output = diffusion_model(
                image, timesteps=torch.Tensor((t,)).to(input_noise.device), context=conditioning
            )
            # print(model_output.shape)
            # 2. compute previous image: x_t -> x_t-1
            noisy_image = image[:, 0, :, :]
            noisy_image = noisy_image[:,None,:,:]
            image, _ = scheduler.step(model_output, t, noisy_image)
            image = torch.cat((image,input_image),1)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        image  = image[:, 0, :, :]
        image = image[:,None,:,:]
        if save_intermediates:
            return image, intermediates
        else:
            return image

class transformDiffusionInfererWithClass(DiffusionInferer):
    """
    Extended DiffusionInferer that supports both context and class_labels as conditioning inputs.
    """
    def __init__(self, scheduler: nn.Module) -> None:
        Inferer.__init__(self)
        self.scheduler = scheduler

    def __call__(
        self,
        inputs: torch.Tensor,
        append_image: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
        noisy_image = torch.cat((noisy_image, append_image), dim=1)
        prediction = diffusion_model(
            x=noisy_image,
            timesteps=timesteps,
            context=context,
            class_labels=class_labels
        )
        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        input_image: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Callable[..., torch.Tensor] | None = None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        verbose: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,  # <---
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if not scheduler:
            scheduler = self.scheduler
        image = torch.cat((input_noise, input_image), dim=1)

        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)

        intermediates = []
        for step, t in enumerate(progress_bar):
            model_output = diffusion_model(
                x=image,
                timesteps=torch.Tensor((t,)).to(input_noise.device),
                context=context,
                class_labels=class_labels
            )
            noisy_image = image[:, 0, :, :].unsqueeze(1)
            image, _ = scheduler.step(model_output, t, noisy_image)
            image = torch.cat((image, input_image), dim=1)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)

            if progress_callback is not None:
                progress_callback(step + 1, len(scheduler.timesteps))
                
        image = image[:, 0, :, :].unsqueeze(1)
        return (image, intermediates) if save_intermediates else image


