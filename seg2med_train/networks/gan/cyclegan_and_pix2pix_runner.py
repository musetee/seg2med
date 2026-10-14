# runners/cyclegan_runner.py
from seg2med_train.networks.model_registry import register_model
from seg2med_train.networks.gan.cyclegan_and_pix2pix import train as train_cycleGAN_pix2pix
from seg2med_train.networks.gan.cyclegan_and_pix2pix import test as test_cycleGAN_pix2pix

import importlib.util
import subprocess
import sys

@register_model('cycle_gan')
@register_model('pix2pix')
class CycleGAN_pix2pix_Runner:
    def __init__(self, opt, paths, train_loader, val_loader,
                 train_patient_IDs, remaining_args=None, **kwargs):
        self.opt = opt
        self.paths = paths
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_patient_IDs = train_patient_IDs
        self.remaining_args = remaining_args or []
        self.train_module = train_cycleGAN_pix2pix
        self.test_module = test_cycleGAN_pix2pix
        
        self._check_and_install_dependencies(['visdom', 'dominate'])

    def _check_and_install_dependencies(self, packages):
        for package in packages:
            if importlib.util.find_spec(package) is None:
                print(f"Installing missing package: {package}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            else:
                print(f"Package '{package}' is already installed.")

    def train(self):
        self.train_module.run_(
            self.remaining_args,
            self.opt,
            self.train_loader,
            self.train_patient_IDs,
            self.paths["root"]
        )

    def test(self):
        self.test_module.run(
            self.remaining_args,
            self.opt,
            self.val_loader,
            self.paths
        )
