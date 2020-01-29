import numpy as np
import random

from base import BaseProposer, BaseResponder


class ThresProposer(BaseProposer):
    """ Deterministic proposer """
    def __init__(self):
        super().__init__()
        self.proposition = np.random.random()

    def propose(self):
        return self.proposition

    def mutate(self, alpha):
        self.proposition = np.clip(self.proposition + alpha*np.random.randn(), 0,1)


class ThresResponder(BaseResponder):
    """ Deterministic responder """
    def __init__(self):
        super().__init__()
        self.accept_bound = np.random.random()

    def respond(self, part):
        return part >= self.accept_bound

    def mutate(self, alpha):
        self.accept_bound = np.clip(self.accept_bound + alpha * np.random.randn(), 0, 1)


class ProbProposer(BaseProposer):
    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins
        self.proposition_probs = np.random.random(bins)
        self._normalize()

    def _normalize(self):
        self.proposition_probs /= np.sum(self.proposition_probs)

    def propose(self):
        r = np.random.random()
        for k in range(self.bins):
            r -= self.proposition_probs[k]
            if r <= 0:
                break

        return (1 / (self.bins-1)) * k

    def mutate(self, alpha):
        self.proposition_probs += np.random.randn(self.bins)
        np.clip(self.proposition_probs, 0,10, out=self.proposition_probs)
        self._normalize()


class ProbResponder(BaseResponder):
    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins
        self.accept_probs = np.random.random(bins)

    def respond(self, part):
        k = int(part * self.bins - 1e-6)
        return np.random.random() < self.accept_probs[k]

    def mutate(self, alpha):
        self.accept_probs += np.random.randn(self.bins)
        np.clip(self.accept_probs, 0,1, out=self.accept_probs)


class Chooser:
    choices = {}
    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins
        self.probs = np.random.random(bins)
        self._normalize()

        if bins not in self.choices:
            self.choices[bins] = np.linspace(0,1, bins)

    def _normalize(self):
        self.probs /= np.sum(self.probs)

    def random_choice(self):
        return np.random.choice(self.choices[self.bins], p=self.probs)

    def mutate(self, alpha:float):
        k = np.random.randint(0, self.bins)
        self.probs[k] += np.random.randn() * alpha
        np.clip(self.probs, 0, 1, out=self.probs)
        self._normalize()


class ChooserProposer(BaseProposer, Chooser):
    def __init__(self, bins=10):
        BaseProposer.__init__(self)
        Chooser.__init__(self, bins)

    def propose(self):
        return self.random_choice()

    def mutate(self, alpha: float):
        Chooser.mutate(self, alpha)

class ChooserResponder(BaseResponder, Chooser):
    def __init__(self, bins=10):
        Chooser.__init__(self, bins)
        BaseProposer.__init__(self)

    def respond(self, part):
        bound = self.random_choice()
        return bound < part

    def mutate(self, alpha: float):
        Chooser.mutate(self, alpha)
