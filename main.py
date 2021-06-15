import torch
import numpy as np
import math
import time
import argparse
from data import get_data
import torch.utils.data as Data
import torch.optim as optim
from AutoRec import AutoRec
from train import train_dann,train_autorec
from DANN import DANN
from test import test_dann,test_auto_rec,test_single_domain
import os




if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    parser = argparse.ArgumentParser(description='AutoRec')
    parser.add_argument('--source_data',choices=['music'],default='music')
    parser.add_argument('--target_data',choices=['book'],default='book')
    parser.add_argument('--embedding_size', type=int, default=500)
    parser.add_argument('--alpha_value', type=float, default=1)

    parser.add_argument('--auto_epoch', type=int, default=50)
    parser.add_argument('--dann_epoch',type=int,default=50)
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--optimizer_method', choices=['Adam', 'RMSProp'], default='Adam')
    parser.add_argument('--grad_clip', type=bool, default=False)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--decay_epoch_step', type=int, default=50, help="decay the learning rate for each n epochs")

    parser.add_argument('--random_seed', type=int, default=1000)
    parser.add_argument('--display_step', type=int, default=1)

    parser.add_argument('--p_hidden',type=int,default=100)
    parser.add_argument('--e_hidden',type=int,default=100)
    parser.add_argument('--c_hidden',type=int,default=50)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--mu', type=float, default=1000)
    parser.add_argument('--lambda_value', type=float, default=0.1)

    parser.add_argument('--test_mode',choices=['single','cross'],default='cross')

    args = parser.parse_args()

    np.random.seed(args.random_seed)



    train_ratio = 0.8

    path = "./data/"

    train_r, train_mask_r, test_r, test_mask_r, user_train_set, item_train_set, user_test_set, \
    item_test_set,num_users,num_items,num_total_ratings = get_data(path,'music',train_ratio)

    ttrain_r, ttrain_mask_r, ttest_r, ttest_mask_r, tuser_train_set, titem_train_set, tuser_test_set, \
    titem_test_set,tnum_users,tnum_items,tnum_total_ratings = get_data(path,'book',train_ratio)

    args.cuda = True
    rec = AutoRec(args, num_users, num_items)
    rect = AutoRec(args,tnum_users,tnum_items)
    if args.cuda:
        rec.cuda()
        rect.cuda()

    optimer = optim.Adam(rec.parameters(), lr=args.base_lr, weight_decay=1e-4)
    optimert = optim.Adam(rect.parameters(), lr=args.base_lr, weight_decay=1e-4)

    num_batch = int(math.ceil(num_users / args.batch_size))
    num_batcht = int(math.ceil(tnum_users / args.batch_size))

    torch_dataset = Data.TensorDataset(torch.from_numpy(train_r), torch.from_numpy(train_mask_r),
                                       torch.from_numpy(train_r))
    t_dataset = Data.TensorDataset(torch.from_numpy(ttrain_r), torch.from_numpy(ttrain_mask_r),
                                       torch.from_numpy(ttrain_r))
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    tloader = Data.DataLoader(
        dataset=t_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    '''Train AutoEncoder for source domain'''
    print('Train AutoEncoder for source domain')
    for epoch in range(args.auto_epoch):
        train_autorec(rec,optimer,loader,train_mask_r,epoch=epoch+1)
        test_auto_rec(rec,test_r,test_mask_r,user_test_set,user_train_set,item_test_set,item_train_set,epoch=epoch+1)


    '''Train AutoEncoder for target domain'''
    print('Train AutoEncoder for target domain')
    for epoch in range(args.auto_epoch):
        train_autorec(rect,optimert,tloader,ttrain_mask_r,epoch=epoch+1)
        test_auto_rec(rect,ttest_r,ttest_mask_r,tuser_test_set,tuser_train_set,titem_test_set,titem_train_set,epoch=epoch+1)

    '''Get embedding'''
    print('encode rating matrix')
    embedding = rec.encoder(torch.from_numpy(train_r).type(torch.FloatTensor).cuda())
    tembedding = rect.encoder(torch.from_numpy(ttrain_r).type(torch.FloatTensor).cuda())

    '''Build dataset for dann'''
    dataset_source = Data.TensorDataset(embedding,torch.from_numpy(train_r),torch.from_numpy(train_mask_r))
    dataset_target = Data.TensorDataset(tembedding,torch.from_numpy(ttrain_r),torch.from_numpy(ttrain_mask_r))
    dataloader_source = Data.DataLoader(
        dataset=dataset_source,
        batch_size=args.batch_size,
        shuffle=True
    )


    dataloader_target = Data.DataLoader(
        dataset=dataset_target,
        batch_size=args.batch_size,
        shuffle=True
    )



    dann = DANN(args,num_items,tnum_items)
    dann.cuda()
    optimizer = optim.Adam(dann.parameters(), lr=0.001)

    '''Train DANN'''
    print('train dann')
    if(args.test_mode=='cross'):
        for epoch in range(args.dann_epoch):
            train_dann(args,dann,dataloader_source,dataloader_target,optimizer,epoch+1)
            test_dann(rec,dann,stest_r=test_r,ttest_r=ttest_r,stest_mask_r=test_mask_r,ttest_mask_r=ttest_mask_r,n_epoch=epoch+1)

    elif(args.test_mode=='single'):
        for epoch in range(args.dann_epoch):
            train_dann(args, dann, dataloader_source, dataloader_target, optimizer, epoch + 1)
            test_single_domain(rect,dann,ttest_r,ttest_mask_r,epoch+1)