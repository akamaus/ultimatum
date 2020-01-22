
class BaseProposer:
    def __init__(self):
        self.fitness = 0

    def propose(self):
        raise NotImplementedError()

    def mutate(self):
        raise NotImplementedError()

class BaseResponder:
    def __init__(self):
        self.fitness = 0

    def respond(self, part):
        raise NotImplementedError()

    def mutate(self):
        raise NotImplementedError()
