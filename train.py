import torch
from torch.autograd import Variable
import numpy as np
from DANN import  DANN


def train_autorec(rec, optimer, loader,train_mask_r,epoch):
    RMSE = 0
    cost_all = 0
    for step, (batch_x, batch_mask_x, batch_y) in enumerate(loader):
        batch_x = batch_x.type(torch.FloatTensor).cuda()
        batch_mask_x = batch_mask_x.type(torch.FloatTensor).cuda()

        decoder = rec(batch_x)
        loss, rmse = rec.loss(decoder=decoder, input=batch_x, optimizer=optimer, mask_input=batch_mask_x)

        optimer.zero_grad()
        loss.backward()
        optimer.step()
        cost_all += loss
        RMSE += rmse

    RMSE = np.sqrt(RMSE.detach().cpu().numpy() / (train_mask_r == 1).sum())
    if(epoch%10==0 or epoch==1) :
        print('epoch ', epoch, ' train RMSE : ', RMSE)

def train_dann(args,dann, dataloader_source,dataloader_target, optimizer, n_epoch):
    dann.train()


    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    RMSE = 0
    pre_LOSS = 0
    domain_LOSS = 0
    NUM = 0

    for i in range(len_dataloader):
        p = float(i+n_epoch*len_dataloader) / args.dann_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10*p)) -1

        data_source = data_source_iter.__next__()
        data_target = data_target_iter.__next__()
        s_embedding, s_label, mask_s = data_source
        t_embedding, t_label, mask_t = data_target


        NUM += (mask_s==1).sum()
        NUM += (mask_t==1).sum()

        dann.zero_grad()
        batch_size = len(s_label)
        domain_label = torch.zeros(batch_size).long()

        #GPU
        s_embedding = s_embedding.cuda()
        s_label = s_label.cuda()
        t_embedding = t_embedding.cuda()
        t_label = t_label.cuda()
        domain_label = domain_label.cuda()
        mask_s = mask_s.cuda()
        mask_t = mask_t.cuda()

        #---------------------------

        s_pre, t_pre, c_pre = dann(input=s_embedding,constant=alpha)

        pre_loss,rmse  = dann.predictor_loss(optimizer=optimizer,s_pre=s_pre,t_pre=t_pre,s_label=s_label,t_label=t_label,\
        mask_s=mask_s,mask_t=mask_t,beta=args.beta,lambda_value=args.lambda_value)

        domain_loss = dann.classifier_loss(c_pre,domain_label)


        total_loss = pre_loss + args.mu*domain_loss
        #print(total_loss)
        total_loss.backward(retain_graph=True)
        optimizer.step()

        pre_LOSS += pre_loss
        domain_LOSS += domain_loss
        RMSE += rmse
        #------------------
        dann.zero_grad()
        batch_size = len(t_label)
        s_pre, t_pre, c_pre = dann(input=t_embedding,constant=alpha)
        domain_label = torch.ones(batch_size).long()
        domain_label = domain_label.cuda()
        pre_loss,_ = dann.predictor_loss(optimizer=optimizer,s_pre=s_pre,t_pre=t_pre,s_label=s_label,t_label=t_label,\
        mask_s=mask_s,mask_t=mask_t,beta=args.beta,lambda_value=args.lambda_value)
        domain_loss = dann.classifier_loss(c_pre,domain_label)
        total_loss = pre_loss + args.mu* domain_loss
        total_loss.backward(retain_graph=True)
        optimizer.step()

        pre_LOSS += pre_loss
        domain_LOSS += domain_loss
        RMSE += rmse

        # RMSE = np.sqrt(RMSE/ NUM.detach().cpu().numpy())


    if(n_epoch%10==0 or n_epoch==1):
        print('epoch:{},preloss:{},domain_loss:{},RMSE:{}'.format(n_epoch,pre_LOSS,domain_LOSS,np.sqrt(RMSE.detach().cpu().numpy()/NUM.detach().cpu().numpy())))

