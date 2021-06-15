import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Extractor(nn.Module):
    def __init__(self,embedding_size,hidden_size):
        super(Extractor,self).__init__()
        self.fn = nn.Sequential(
            nn.Linear(embedding_size,hidden_size)
        )

    def forward(self,input):
        return self.fn(input)

class Predictor(nn.Module):
    def __init__(self,input_size,hidden,snum_item,tnum_item):
        super(Predictor,self).__init__()
        self.spredictor = nn.Sequential(
            nn.Linear(input_size,hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden,hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden,snum_item)
        )

        self.tpredictor = nn.Sequential(
            nn.Linear(input_size,hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden,hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden,tnum_item)
        )

    def forward(self,input):
        s_pre = self.spredictor(input)
        t_pre = self.tpredictor(input)
        return  s_pre,t_pre


class Domain_classifier(nn.Module):
    def __init__(self,input_size,hidden):
        super(Domain_classifier,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )


    def forward(self,input,constant):
        input = GradReverse.grad_reverse(input,constant)
        return self.fc(input)



class DANN(nn.Module):
    def __init__(self,args,snum_item,tnum_item):
        super(DANN, self).__init__()
        self.extractor = Extractor(args.embedding_size,args.e_hidden)
        self.predictor = Predictor(args.e_hidden,args.p_hidden,snum_item,tnum_item)
        self.classifier = Domain_classifier(args.e_hidden,args.c_hidden)

    def forward(self,input,constant):
        feature = self.extractor(input)
        s_pre, t_pre = self.predictor(feature)
        c_pre = self.classifier(feature,constant)
        return s_pre, t_pre, c_pre

    def predictor_loss(self,optimizer,s_pre,t_pre,s_label,t_label,mask_s,mask_t,beta,lambda_value):
        rmse  = 0
        reg = 0

        rmse += ((s_pre-s_label)*mask_s).pow(2).sum()

        rmse += beta*(((t_pre-t_label)*mask_t).pow(2).sum())

        for i in optimizer.param_groups:
            for j in i['params']:
                if j.data.dim() == 2  or j.data.dim() == 1:
                    reg += torch.t(j.data).pow(2).sum()  # 正则化项

        loss = rmse + lambda_value*reg
        return loss, rmse

    def classifier_loss(self,c_pre,domain_label):
        loss_func = nn.CrossEntropyLoss()

        return loss_func(c_pre,domain_label)

