import torch
import torch.nn as nn


class AutoRec(nn.Module):
    def __init__(self,args,num_users,num_items):
        super(AutoRec,self).__init__()

        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = args.embedding_size
        self.alpha_value = args.alpha_value

        self.encoder = nn.Sequential(
            nn.Linear(self.num_items,self.embedding_size),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_size,self.num_items),

        )

    def forward(self,torch_input):
        encoder = self.encoder(torch_input)
        decoder = self.decoder(encoder)
        return decoder

    def loss(self,decoder,input,optimizer,mask_input):
        cost  = 0

        temp2 = 0

        cost += ((decoder - input) * mask_input).pow(2).sum()
        rmse = cost

        for i in optimizer.param_groups:
            for j in i['params']:
                # print(type(j.data), j.shape,j.data.dim())
                if j.data.dim() == 2 or j.data.dim()==1:
                    temp2 += torch.t(j.data).pow(2).sum()  # 正则化项

        cost += temp2 * self.alpha_value * 0.5
        return cost, rmse







