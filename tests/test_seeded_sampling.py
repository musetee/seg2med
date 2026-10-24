"""Seeded DDPM sampling: named noise streams independent of batching (P-006)."""

import hashlib
import unittest

import torch
from generative.networks.schedulers import DDPMScheduler

from seg2med_train.networks.ddpm.seeded_sampling import NoiseStreams, seeded_sample, stream_seed


def reference_seed(*parts):
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2**63


def reference_randn(shape, *parts):
    return torch.randn(shape, dtype=torch.float32, generator=torch.Generator().manual_seed(reference_seed(*parts)))


class HalfModel(torch.nn.Module):
    """Deterministic stand-in for the UNet; returns fp16 like autocast inference."""

    def forward(self, x, timesteps, context=None):
        return (0.3 * x[:, :1] + 0.2 * x[:, 1:2] - 0.1 * x[:, 2:3] + 0.001 * timesteps.float().mean()).half()


class SeededSamplingTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = DDPMScheduler(num_train_timesteps=10, schedule="linear_beta",
                                       beta_start=0.0001, beta_end=0.02)
        self.scheduler.set_timesteps(num_inference_steps=10)
        conditioning = torch.linspace(0, 1, 4 * 2 * 4 * 5).reshape(4, 2, 4, 5)
        self.conditioning = conditioning

    def streams(self, shared):
        return NoiseStreams("P006-v1", "006", "006_BL-FU", 2024, shared=shared)

    def test_stream_seed_matches_public_contract_and_manifest_values(self):
        self.assertEqual(stream_seed("P006-v1", "006", "006_BL-FU", 2024),
                         reference_seed("P006-v1", "006", "006_BL-FU", "2024"))
        # Leading zeros of patient ids are kept; integers are plain decimal text.
        self.assertNotEqual(stream_seed("P006-v1", "006"), stream_seed("P006-v1", "6"))
        self.assertEqual(stream_seed("a", 7), stream_seed("a", "7"))
        # Value recorded for patient 006 in logs/p006_inputs/selected_volumes_v1.csv.
        self.assertEqual(stream_seed("P006-v1", "006", "006_BL-FU_BL_linear_to_FU_wmh-synthseg", 2024),
                         7290947601639052552)
        with self.assertRaises(ValueError):
            stream_seed("a|b")

    def test_initial_noise_per_slice_or_shared_from_slice_zero(self):
        independent = self.streams(False).initial_noise([0, 3], (4, 5))
        self.assertEqual(independent.dtype, torch.float32)
        self.assertEqual(tuple(independent.shape), (2, 1, 4, 5))
        for row, z in enumerate((0, 3)):
            expected = reference_randn((1, 1, 4, 5), "P006-v1", "init", "006", "006_BL-FU", "2024", str(z))
            self.assertTrue(torch.equal(independent[row:row + 1], expected))
        shared = self.streams(True).initial_noise([2, 3], (4, 5))
        self.assertTrue(torch.equal(shared[0], independent[0]))
        self.assertTrue(torch.equal(shared[1], independent[0]))

    def test_step_noise_does_not_depend_on_strategy(self):
        first = torch.randn((1, 1, 4, 5), generator=self.streams(True).step_generator(3, 7))
        second = torch.randn((1, 1, 4, 5), generator=self.streams(False).step_generator(3, 7))
        expected = reference_randn((1, 1, 4, 5), "P006-v1", "step", "006", "006_BL-FU", "2024", "3", "7")
        self.assertTrue(torch.equal(first, expected))
        self.assertTrue(torch.equal(second, expected))

    def test_sampling_is_fp32_and_independent_of_batch_split(self):
        model = HalfModel()
        streams = self.streams(False)
        whole = seeded_sample(model, self.scheduler, self.conditioning, streams, [0, 1, 2, 3])
        parts = torch.cat([seeded_sample(model, self.scheduler, self.conditioning[i:i + 1], streams, [i])
                           for i in range(4)])
        self.assertEqual(whole.dtype, torch.float32)
        self.assertTrue(torch.equal(whole, parts))
        # Reference loop: unchanged scheduler.step, fp32 inputs, one generator per (z, t).
        reference = []
        for i in range(4):
            image = reference_randn((1, 1, 4, 5), "P006-v1", "init", "006", "006_BL-FU", "2024", str(i))
            for t in self.scheduler.timesteps:
                output = model(torch.cat((image, self.conditioning[i:i + 1]), 1),
                               timesteps=torch.Tensor((t,))).float()
                generator = torch.Generator().manual_seed(
                    reference_seed("P006-v1", "step", "006", "006_BL-FU", "2024", str(i), str(int(t))))
                image, _ = self.scheduler.step(output, t, image, generator=generator)
            reference.append(image)
        self.assertTrue(torch.equal(whole, torch.cat(reference)))

    def test_sampling_keeps_no_autograd_graph(self):
        # A graph kept across 500 steps exhausted 48 GB in the first real run.
        class Learned(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 1)

            def forward(self, x, timesteps, context=None):
                return self.conv(x)

        output = seeded_sample(Learned(), self.scheduler, self.conditioning, self.streams(True), [0, 1, 2, 3])
        self.assertFalse(output.requires_grad)

    def test_strategies_differ_only_in_initial_noise(self):
        model = HalfModel()
        shared = seeded_sample(model, self.scheduler, self.conditioning, self.streams(True), [0, 1, 2, 3])
        independent = seeded_sample(model, self.scheduler, self.conditioning, self.streams(False), [0, 1, 2, 3])
        # Slice 0 uses the same init stream in both strategies.
        self.assertTrue(torch.equal(shared[0], independent[0]))
        self.assertFalse(torch.equal(shared[1], independent[1]))


if __name__ == "__main__":
    unittest.main()
