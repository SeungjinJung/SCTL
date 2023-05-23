from torch import optim

# Optimizer for Temperature Scalinig, Ensemble Temperature Scaling, and Class-based Temperature Scaling
def lbfgs(cal, args):
    return [optim.LBFGS(cal.parameters(), lr=args.lr, max_iter=args.n_iter)]

# Optimizer for Temperature Scalinig, Ensemble Temperature Scaling, and Class-based Temperature Scaling, by using Focal Loss or Label Smoothing
def lbfgs_schedule(cal, _):
    return [optim.LBFGS(cal.parameters(), lr=0.005, max_iter=200), 
            optim.LBFGS(cal.parameters(), lr=0.003, max_iter=400), 
            optim.LBFGS(cal.parameters(), lr=0.001, max_iter=400)]

# Optimizer for Parameterized Temperature Scalinig
def adam(cal, args):
    return [[optim.Adam(cal.parameters(), lr=args.lr),args.n_iter]]

# Optimizer for Parameterized Temperature Scalinig, by using Focal Loss or Label Smoothing
def adam_schedule(cal, _):
    return [[optim.Adam(cal.parameters(), lr=0.005), 200],
            [optim.Adam(cal.parameters(), lr=0.003), 400], 
            [optim.Adam(cal.parameters(), lr=0.001), 400]]
