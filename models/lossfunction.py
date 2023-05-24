import torch
import numpy as np
from scipy import optimize
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# Baseline Loss
class cross_entropy_loss(nn.Module):
    def __init__(self, args):
        super(cross_entropy_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.loss(input, target)
    
class label_smoothing_loss(nn.Module):
    def __init__(self, args):
        super(label_smoothing_loss, self).__init__()
        args.optim += '_schedule'
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.05).cuda()

    def forward(self, input, target):
        return self.loss(input, target)

class focal_loss(nn.Module):
    def __init__(self, args, gamma=3, alpha=None, size_average=True):
        super(focal_loss, self).__init__()
        args.optim += '_schedule'
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()
    
class scaling_classwise_training_loss(nn.Module):
    def __init__(self, args):
        super(scaling_classwise_training_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.n_class = args.n_class
        self.norm = args.norm
        self.step = 0
        self.alpha = 1
        self.beta = 1.5

    def forward(self, input, target):
        losses = torch.zeros(self.n_class)
        for i in range(self.n_class):
            indice = target.eq(i)
            tx = input[indice]
            ty = target[indice]
            losses[i] = self.loss(tx, ty)

        loss = losses.clone().detach()

        if self.norm == 'ND':
            norm = (loss-loss.mean())/loss.std() 
        elif self.norm == 'MM': 
            norm = (loss-loss.min())/(loss.max()-loss.min())
        elif self.norm == 'CM':
            norm = (loss-loss.mean())/(loss.max()-loss.min())

        if self.step == 0:
            self.first = loss.tolist()

        # Optimize alpha and beta
        elif self.step == 1:
            self.optim = False
            self.optimize_scailing_estimator(norm.tolist(), loss.tolist())

        self.step += 1
                
        # scale loss
        losses *= self.scailing_estimator(norm)
        return losses.sum()
    
    def scailing_estimator(self, x):
        return self.beta/(1+np.exp(-x/self.alpha)) - self.beta/2 + 1
    
    def optimize_scailing_estimator(self, norm, loss):
        def func(x, *args):
            return np.sqrt(((np.array(args[2]) - (x[1]/(1+np.exp(-np.array(args[0])/x[0])) -x[1]/2) * (np.array(args[1])-np.array(args[2])) - np.array(args[1]).mean()) ** 2).sum())
        
        opt = optimize.minimize(func, (self.alpha, self.beta), args=(norm, loss, self.first, self.n_class), method='SLSQP', 
                                bounds=((0.1, np.log(self.n_class)/2),(1.5, 2.0)), options={'disp':False})

        self.alpha, self.beta = opt.x
