"""Reproducible DDPM sampling from named, per-slice noise streams.

Every random draw comes from its own CPU generator seeded by a public tuple, so
results do not depend on batch grouping, device or global RNG state:

* initial noise of slice z:  (namespace, "init", patient, volume, seed_label, z)
  (shared strategy: every slice reuses the stream of z=0)
* DDPM step noise at (z, t): (namespace, "step", patient, volume, seed_label, z, t)
  identical for both strategies, so they differ only in the initial noise.

Seed = first 8 bytes of SHA256(UTF-8("|".join(parts))), unsigned big-endian, mod 2**63.
The scheduler's own step() formula is used unchanged; model output and sample are
cast to float32 before each step, so noise and updates are float32.
"""

import contextlib
import hashlib

import torch


def stream_seed(*parts):
    texts = [str(part) for part in parts]
    if any("|" in text for text in texts):
        raise ValueError(f"Seed stream parts must not contain '|': {texts}")
    digest = hashlib.sha256("|".join(texts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2**63


def _generator(*parts):
    return torch.Generator().manual_seed(stream_seed(*parts))


class NoiseStreams:
    def __init__(self, namespace, patient_id, volume_id, seed_label, shared, shared_z=0):
        self.prefix = (namespace,)
        self.volume = (str(patient_id), str(volume_id), str(seed_label))
        self.shared, self.shared_z = bool(shared), int(shared_z)

    def initial_noise(self, z_indices, shape):
        """(len(z_indices), 1, *shape) float32 CPU standard normal noise."""
        planes = []
        for z in z_indices:
            stream_z = self.shared_z if self.shared else int(z)
            generator = _generator(*self.prefix, "init", *self.volume, stream_z)
            planes.append(torch.randn((1, 1, *shape), dtype=torch.float32, generator=generator))
        return torch.cat(planes)

    def step_generator(self, z, t):
        return _generator(*self.prefix, "step", *self.volume, int(z), int(t))


@torch.no_grad()
def seeded_sample(model, scheduler, conditioning_input, streams, z_indices, context=None,
                  forward_context=contextlib.nullcontext):
    """DDPM sampling of one slice batch; returns the float32 generated images.

    conditioning_input: (B, C, H, W) prepared conditioning channels; z_indices gives
    the original slice index of each batch row. forward_context wraps only the
    network forward (e.g. autocast).
    """
    if len(z_indices) != conditioning_input.shape[0]:
        raise ValueError("Need one slice index per batch row.")
    device = conditioning_input.device
    image = streams.initial_noise(z_indices, conditioning_input.shape[2:]).to(device)
    for t in scheduler.timesteps:
        with forward_context():
            output = model(torch.cat((image, conditioning_input), 1),
                           timesteps=torch.Tensor((t,)).to(device), context=context)
        output = output.float()
        image = torch.cat([
            scheduler.step(output[row:row + 1], t, image[row:row + 1],
                           generator=streams.step_generator(z, t))[0]
            for row, z in enumerate(z_indices)
        ])
    return image
