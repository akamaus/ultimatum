import numpy as np

from base import BaseProposer, BaseResponder

class ThresProposer(BaseProposer):
    def __init__(self):
        super().__init__()
        self.proposition = np.random.random()

    def propose(self):
        return self.proposition

    def mutate(self, alpha):
        self.proposition = np.clip(self.proposition + alpha*np.random.randn(), 0,1)


class ThresResponder(BaseResponder):
    def __init__(self):
        super().__init__()
        self.accept_bound = np.random.random()

    def respond(self, part):
        return part >= self.accept_bound

    def mutate(self, alpha):
        self.accept_bound = np.clip(self.accept_bound + alpha * np.random.randn(), 0, 1)