import torch
is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx]
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)

class MixSampler(torch.utils.data.sampler.Sampler):
    """Shuffled epochs with a fixed real:synthetic mix (PLAN.md stage B).

    Each epoch has num_samples indices: round(real_fraction * num_samples) drawn from the real
    images and the rest from the synthetic ones, each pool cycled in a fresh random order (so an
    image repeats only after its whole pool has been used), then shuffled together.
    BalancedBatchSampler above is the 0.5 case but with a fixed, unshuffled order every epoch.
    labels: per-index synthetic flag (truthy = synthetic).
    """
    def __init__(self, labels, real_fraction, num_samples):
        self.real = [i for i, s in enumerate(labels) if not s]
        self.syn = [i for i, s in enumerate(labels) if s]
        self.num_samples = num_samples
        self.n_real = int(round(real_fraction * num_samples))

    @staticmethod
    def _draw(pool, n):
        out = []
        while len(out) < n:
            p = pool[:]
            random.shuffle(p)
            out += p
        return out[:n]

    def __iter__(self):
        idx = self._draw(self.real, self.n_real) + self._draw(self.syn, self.num_samples - self.n_real)
        random.shuffle(idx)
        return iter(idx)

    def __len__(self):
        return self.num_samples
