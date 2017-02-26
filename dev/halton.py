import ghalton

class Halton:
    def __init__(self, ranges):
        self.num_dim = len(ranges)
        self.lower, self.upper = zip(*ranges)
        self.sequencer = ghalton.Halton(self.num_dim)
        return None
    def Get(self, num_points=1):
        point = self.sequencer.get(1)[0]
        scaled_point = []
        for i in range(self.num_dim):
            scaled_point.append(point[i]*(self.upper[i] - self.lower[i]) + self.lower[i])
        return scaled_point