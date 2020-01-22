import numpy as np
import copy

import strategies as S


class Population:
    def __init__(self, size, mutation_strength=0.05, culling_ratio=0.1):
        self.size = size
        self.proposers = [ S.ThresProposer() for _ in range(size) ]
        self.responders = [ S.ThresResponder() for _ in range(size) ]

        self.culling_ratio = culling_ratio
        self.mutation_strength = mutation_strength

    def evolve(self, n_rounds=10):
        for k in range(n_rounds):
            perm = np.random.permutation(self.size)
            for i,j in enumerate(perm):
                self.play(self.proposers[i], self.responders[j])

        self.natural_selection(self.proposers)
        self.natural_selection(self.responders)

    def natural_selection(self, population):
        """ Let worst part of population die, breed survivals until size is restored """
        population.sort(key=lambda x: x.fitness, reverse=True)
        for k in range(int(len(population) * self.culling_ratio)):
            population.pop()

        breeds = np.random.randint(0, len(population), self.size - len(population))
        for bi in breeds:
            child = copy.deepcopy(population[bi])
            child.mutate(self.mutation_strength)
            population.append(child)

        assert len(population) == self.size

    def play(self, proposer, responder):
        prop = proposer.propose()
        assert 0 <= prop <= 1
        resp = responder.respond(prop)
        if resp:
            proposer.fitness += 1 - prop
            responder.fitness += prop
        return resp