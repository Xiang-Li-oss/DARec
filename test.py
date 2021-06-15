import torch
from torch import nn
import numpy as np

def test_auto_rec(rec,test_r,test_mask_r,user_test_set,user_train_set,item_test_set,item_train_set,epoch):
    test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor).cuda()
    test_mask_r_tensor = torch.from_numpy(test_mask_r).type(torch.FloatTensor).cuda()
    # test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor)
    # test_mask_r_tensor = torch.from_numpy(test_mask_r).type(torch.FloatTensor)

    decoder = rec(test_r_tensor)
    decoder = torch.from_numpy(np.clip(decoder.detach().cpu().numpy(),a_min=1,a_max=5)).cuda()

    unseen_user_test_list = list(user_test_set - user_train_set)
    unseen_item_test_list = list(item_test_set - item_train_set)

    for user in unseen_user_test_list:
        for item in unseen_item_test_list:
            if test_mask_r[user, item] == 1:  # 如果在测试集中存在这条评分记录，则进行记录decoder[user,item]=3
                decoder[user, item] = 3

    mse = ((decoder - test_r_tensor) * test_mask_r_tensor).pow(2).sum()
    RMSE = mse.detach().cpu().numpy() / (test_mask_r == 1).sum()
    RMSE = np.sqrt(RMSE)

    if(epoch%10==0 or epoch==1):
        print('epoch ', epoch, ' test RMSE : ', RMSE)

def test_dann(autorec,dann,stest_r,ttest_r,stest_mask_r,ttest_mask_r,n_epoch):
    stest_r_tensor = torch.from_numpy(stest_r).type(torch.FloatTensor).cuda()
    stest_mask_tensor = torch.from_numpy(stest_mask_r).type(torch.FloatTensor).cuda()
    ttest_r_tensor = torch.from_numpy(ttest_r).type(torch.FloatTensor).cuda()
    ttest_mask_tensor = torch.from_numpy(ttest_mask_r).type(torch.FloatTensor).cuda()



    embedding = autorec.encoder(stest_r_tensor)
    _,output,_ = dann(embedding,1)
    output = torch.from_numpy(np.clip(output.detach().cpu().numpy(),a_min=1,a_max=5)).cuda()
    mse = ((output-ttest_r_tensor)*ttest_mask_tensor).pow(2).sum()
    RMSE = mse.detach().cpu().numpy() / (ttest_mask_r == 1).sum()
    RMSE = np.sqrt(RMSE)

    if(n_epoch%10==0 or n_epoch==1):
        print('epoch:{},test rmse:{}'.format(n_epoch,RMSE))

def test_single_domain(autorec,dann,ttest_r,ttest_mask_r,epoch):
    ttest_r_tensor = torch.from_numpy(ttest_r).type(torch.FloatTensor).cuda()
    ttest_mask_r_tensor = torch.from_numpy(ttest_mask_r).type(torch.FloatTensor).cuda()

    embedding = autorec.encoder(ttest_r_tensor)
    _,output,_ = dann(embedding,1)
    output = torch.from_numpy(np.clip(output.detach().cpu().numpy(),a_min=1,a_max=5)).cuda()
    se = ((output-ttest_r_tensor)*ttest_mask_r_tensor).pow(2).sum()
    mse = se.detach().cpu().numpy() / (ttest_mask_r==1).sum()
    rmse = np.sqrt(mse)

    if(epoch%10==0 or epoch==1):
        print('epoch:{},test rmse:{}'.format(epoch,rmse))