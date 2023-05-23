import torch
from torch import nn

class temperature_scaler(nn.Module):
    def __init__(self, args):
        super(temperature_scaler, self).__init__()
        args.optim = 'lbfgs'
        # Call a Tmeperature Scaling parameter.
        self.t = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x / self.t
    
class ensemble_scaler(nn.Module):
    def __init__(self, args):
        super(ensemble_scaler, self).__init__()
        # Call a Tmeperature Scaling parameter.
        self.w = nn.Parameter(torch.tensor((1.0, 0.0, 0.0)))

    def forward(self, x1, x2, x3):
        return x1*self.w[0] + x2*self.w[1] + x3*self.w[2]
    
class ensemble_temperature_scaler(nn.Module):
    def __init__(self, args):
        super(ensemble_temperature_scaler, self).__init__()
        self.n_class = args.n_class

        self.t = temperature_scaler(args)
        self.w = ensemble_scaler(args)

    def forward(self, x):
        return self.w(self.t(x), x, 1/self.n_class)
    
    
class parameterized_temperature_scaler(nn.Module):
    def __init__(self, args):
        super(parameterized_temperature_scaler, self).__init__()
        args.optim = 'adam'
        
        for i in range(4):
            if i == 0:
                model = [nn.Sequential(nn.Linear(10,2),nn.ReLU())]
            else:
                model += [nn.Sequential(nn.Linear(2,2),nn.ReLU())]
        model += [nn.Linear(2,1)]
        self.models = nn.Sequential(*model)

    def forward(self, x):
        t,_ = x.clone().detach().sort(descending=True)
        t = t[:,:10]
        t = self.models(t)
        return x/t
    
class class_based_temperature_scaler(nn.Module):
    def __init__(self, args):
        super(class_based_temperature_scaler, self).__init__()
        args.optim = 'lbfgs'
        self.T = nn.Parameter(torch.ones(args.n_class))

    def forward(self, x):
        return x / self.T
