import numpy as np
import copy

import strategies as S
from collections import Counter


class Population:
    def __init__(self, mutation_strength=0.05, culling_ratio=0.1, discount=0.9):
        self.proposers = []
        self.responders = []

        self.culling_ratio = culling_ratio
        self.mutation_strength = mutation_strength
        self.discount = discount

        self.prop_mean_log = []
        self.prop_std_log = []
        self.proposer_counters = []
        self.responder_counters = []

    def add_strategy(self, strategy_class, n):
        for i in range(n):
            s = strategy_class()
            if isinstance(s, S.BaseProposer):
                self.proposers.append(s)
            elif isinstance(s, S.BaseResponder):
                self.responders.append(s)

    def evolve(self, n_rounds=10):
        props = []
        for k in range(n_rounds):
            rs = np.random.choice(self.responders, size=len(self.proposers))
            for p,r in zip(self.proposers, rs):
                prop_value, resp = self._play(p, r)
                props.append(prop_value)

            ps = np.random.choice(self.proposers, size=len(self.responders))
            for p,r in zip(ps, self.responders):
                prop_value, resp = self._play(p, r)
                props.append(prop_value)

        self._natural_selection(self.proposers)
        self._natural_selection(self.responders)

        self._zero_fitness(self.proposers)
        self._zero_fitness(self.responders)

        # calculate stats
        self.prop_mean_log.append(np.mean(props))
        self.prop_std_log.append(np.std(props))

        self.proposer_counters.append(self._count_classes(self.proposers))
        self.responder_counters.append(self._count_classes(self.responders))

    @staticmethod
    def _zero_fitness(population):
        for ind in population:
            ind.fitness = 0

    @staticmethod
    def _count_classes(lst):
        cnt = Counter()
        for obj in lst:
            cnt[obj.__class__.__name__] += 1

        return cnt

    def _natural_selection(self, population):
        """ Let worst part of population die, breed survivals until size is restored """

        size = len(population)
        population.sort(key=lambda x: x.fitness, reverse=True)
        for k in range(int(size * self.culling_ratio)):
            population.pop()

        breeds = np.random.randint(0, len(population), size - len(population))
        for bi in breeds:
            child = copy.deepcopy(population[bi])
            child.mutate(self.mutation_strength)
            population.append(child)

        assert len(population) == size

    @staticmethod
    def _play(proposer, responder):
        prop = proposer.propose()
        assert 0 <= prop <= 1
        resp = responder.respond(prop)
        if resp:
            proposer.fitness += 1 - prop
            responder.fitness += prop
        return prop, resp
