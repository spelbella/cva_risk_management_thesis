class Path():
    def __init__(self, t_s, lambdas, r, CVA, Q_s, Swaps, Swaptions, K = None):
        self.t_s = t_s
        self.lambdas = lambdas
        self.r = r
        self.CVA = CVA
        self.Q_s = Q_s
        self.Swaps = Swaps
        self.Swaptions = Swaptions
        self.K = K
