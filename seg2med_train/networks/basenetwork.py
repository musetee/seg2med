from abc import ABC, abstractmethod
class ABCNetwork(ABC):
    @abstractmethod
    def __init__(self):
        """Subclass must implement this method."""
        pass

    def train(self):
        """Subclass must implement this method."""
        pass

    def test(self):
        """Subclass must implement this method."""
        pass

    def postprocess(self):
        """Subclass must implement this method."""
        pass

    def plot_learning_curves(self):
        """Subclass must implement this method."""
        pass

    def plot_metrics(self):
        """Subclass must implement this method."""
        pass

class basenetwork(ABCNetwork):
    def __init__(self):
        pass
    def train(self,train_loader,val_loader):
        pass