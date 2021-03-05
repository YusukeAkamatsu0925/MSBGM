#!/usr/bin/env python
# coding: utf-8

# author:YusukeAkamatsu0925


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler as ss
from numpy import random, matlib, linalg
from PIL import Image
import itertools
class pycolor:
    RED = '\033[31m'
    END = '\033[0m'


# # Settings


num_sub = 5 # number of subjects
num_multi = 3 # number of multi-view features (fMRI activity, visual features, and semantic features)
maxiter_sub = 5 # update number of MSBGM
maxiter = 10 # update number of MVBGM
thres_a_inv = 1e-1 # ARD parameter
delta = 0.6 # trade-off parameter between visual features and semantic features
N_trial = 10 # trial numbers of model training and prediction (N_trial was set to 100 in the original paper.)
Dy = 2500 # dimension of shared latent variable y
Dz = 300 # dimension of shared latent variable z


# # Dataset


Visual_semantic = loadmat('../data/visual&semantic.mat')
v_candidate = Visual_semantic['VGG19_candidate'].T
c_candidate = Visual_semantic['word2vec_candidate'].T
def dataset(subject):
    fMRI = loadmat('../data/subject0{}.mat'.format(subject))
    f_train = fMRI['sub0{}_train_sort'.format(subject)].T
    f_test = fMRI['sub0{}_test_ave'.format(subject)].T
    v_train = Visual_semantic['VGG19_train_sort'].T
    c_train = Visual_semantic['word2vec_train_sort'].T
    return f_train,v_train,c_train,f_test


# # Multi-subject fMRI data


def multi_data():
    D_sub = [0]*num_sub
    X_sub_mean = [0]*num_sub
    X_sub_norm = [0]*num_sub
    X_sub = [0]*num_sub
    for sub in range(num_sub):
        fMRI = loadmat('../data/subject0{}.mat'.format(sub+1))
        fMRI_sub = fMRI['sub0{}_train_sort'.format(sub+1)]
        N = fMRI_sub.shape[0]
        D_sub[sub] = fMRI_sub.shape[1]
        X_sub_mean[sub] = np.mean(fMRI_sub,axis=0)
        X_sub_norm[sub] = np.std(fMRI_sub,axis=0,ddof=1)
        X_sub[sub] = (fMRI_sub.T-matlib.repmat(X_sub_mean[sub],N,1).T)/matlib.repmat(X_sub_norm[sub],N,1).T
    return X_sub,D_sub,N


# # Parameters


def parameter(f_train,v_train,c_train,f_test):
    N = f_train.shape[1]
    N_test = f_test.shape[1]
    D_k = [f_train.shape[0],v_train.shape[0],c_train.shape[0]]
    return N,N_test,D_k


# # Normalize


def normalize(f_train,v_train,c_train):
    N = np.size(f_train,1)
    X_mean = [np.mean(f_train,axis=1),np.mean(v_train,axis=1),np.mean(c_train,axis=1)]
    X_norm = [np.std(f_train,axis=1,ddof=1),np.std(v_train,axis=1,ddof=1),np.std(c_train,axis=1,ddof=1)]
    X_train = [f_train-matlib.repmat(X_mean[0],N,1).T,v_train-matlib.repmat(X_mean[1],N,1).T,c_train-matlib.repmat(X_mean[2],N,1).T]
    X = [X_train[0]/matlib.repmat(X_norm[0],N,1).T,X_train[1]/matlib.repmat(X_norm[1],N,1).T,X_train[2]/matlib.repmat(X_norm[2],N,1).T]
    return X,X_mean,X_norm
def normalize_item(item,X_mean,X_norm):
    N_item = np.size(item,1)
    item = item-matlib.repmat(X_mean,N_item,1).T
    item = item/matlib.repmat(X_norm,N_item,1).T
    return item
def renormalize_item(item,X_mean,X_norm):
    N_item = np.size(item,1)
    item = item*matlib.repmat(X_norm,N_item,1).T
    item = item+matlib.repmat(X_mean,N_item,1).T
    return item


# # Calculate posterior distribution


def predict_multi_sub(W_sub,WW_sub,beta_inv_sub,Dy,f_test,subject):
    # calculate posterior z from fMRI activity
    SigmaZnew = (1/beta_inv_sub[subject])*WW_sub[subject]+np.eye(Dy)
    SigmaZnew_inv = linalg.inv(SigmaZnew)
    prZ_sub = SigmaZnew_inv@((1/beta_inv_sub[subject])*W_sub[subject].T@f_test)
    return prZ_sub


# # Initialize


def initialize(X,D,Dz,num):   
    # Z
    Z = random.randn(Dz,N)
    SigmaZ_inv = np.eye(Dz)
    SZZ = Z@Z.T + N*SigmaZ_inv
    # initialize
    SZZrep, A_inv, A0_inv, gamma0, gamma, gamma_xx, gamma_beta, beta_inv = [0]*num, [0]*num, [0]*num, [0]*num, [0]*num, [0]*num, [0]*num, [0]*num
    for i in range(num):
        SZZrep[i] = matlib.repmat(np.diag(SZZ),D[i],1)
        # alpha,gamma
        A_inv[i] = np.ones((D[i],Dz))
        A0_inv[i] = np.zeros((D[i],Dz))
        gamma0[i] = np.zeros((D[i],Dz))
        gamma[i] = 1/2+gamma0[i]
        gamma_xx[i] = np.sum(X[i]**2)/2
        gamma_beta[i] = D[i]*N/2
        # beta
        beta_inv[i] = 1
    return Z,SZZrep,A_inv,A0_inv,gamma0,gamma,gamma_xx,gamma_beta,beta_inv


# # Update


def update(X,D,Dz,num,maxiter,Z,SZZrep,A_inv,A0_inv,gamma0,gamma,gamma_xx,gamma_beta,beta_inv):
    # initialize
    SigmaW_inv, W, WW, beta_inv_gamma, a_inv, a_inv_max, ix_a = [0]*num, [0]*num, [0]*num, [0]*num, [0]*num, [0]*num, [0]*num
    
    print ('*****************************Dy={},trial={},iteration={}'.format(Dy,t,maxiter))
    for l in range(maxiter):
        # W-step
        for i in range(num):
            SigmaW_inv[i] = A_inv[i]/((1/beta_inv[i])*SZZrep[i]*A_inv[i]+1)
            W[i] = (1/beta_inv[i])*X[i]@Z.T*SigmaW_inv[i]
            WW[i] = np.diag(SigmaW_inv[i].sum(axis=0))+W[i].T@W[i]
        # Z-step
        SigmaZ = 0
        for i in range(num):
            SigmaZ += (1/beta_inv[i])*WW[i]
        SigmaZ += np.eye(Dz)
        SigmaZ_inv = linalg.inv(SigmaZ)
        Z = 0
        for i in range(num):
            Z += (1/beta_inv[i])*SigmaZ_inv@W[i].T@X[i]
        SZZ = Z@Z.T + N*SigmaZ_inv
        # others
        for i in range(num):
            SZZrep[i] = matlib.repmat(np.diag(SZZ),D[i],1)
            # alpha-step
            A_inv[i] = (W[i]**2/2+SigmaW_inv[i]/2+gamma0[i]*A0_inv[i])/gamma[i]
            # beta-step
            beta_inv_gamma[i] = gamma_xx[i]-np.trace(W[i]@Z@X[i].T)+np.trace(SZZ@WW[i])/2
            beta_inv[i] = beta_inv_gamma[i]/gamma_beta[i]
            # find irrelevance parameters
            a_inv[i] = A_inv[i].sum(axis=0)
            a_inv_max[i] = max(a_inv[i])
            ix_a[i] = a_inv[i]>a_inv_max[i]*thres_a_inv
        ix_z = np.logical_and(ix_a[0],ix_a[1])
    #print('Effect number of dimensions (ARD) : {}'.format(np.sum(ix_z)))
    return W,WW,beta_inv,Z


# # Predict


def predict(W,WW,beta_inv,Dz,prZ_sub):
    # calculate posterior z from fMRI activity
    SigmaZnew = (1/beta_inv[0])*WW[0]+np.eye(Dz)
    SigmaZnew_inv = linalg.inv(SigmaZnew)
    prZ = SigmaZnew_inv@((1/beta_inv[0])*W[0].T@prZ_sub)
    # predictive distribution
    v_pred = W[1]@prZ
    v_pred_cov = W[1]@SigmaZnew_inv@W[1].T+beta_inv[1]*np.eye(D[1])
    c_pred = W[2]@prZ
    c_pred_cov = W[2]@SigmaZnew_inv@W[2].T+beta_inv[2]*np.eye(D[2])
    return v_pred,c_pred


# # Estimate image categories


def evaluate(V_pred,C_pred):
    # Estimate image categories from visual features
    v_corr = (1 - cdist(V_pred.T, v_candidate.T, metric='correlation'))
    # Estimate image categories from category features
    c_corr = (1 - cdist(C_pred.T, c_candidate.T, metric='correlation'))
    # Rankings of estimated image categories
    def calc_rank(corr):
        sort = np.sort(corr,axis=1)[:,::-1]
        sort_ix = np.argsort(corr,axis=1)[:,::-1]
        Rank = []
        for i in range(N_test):
            Rank.append(int(np.where(sort_ix[i,:]==i)[0]+1))
        return Rank,sort,sort_ix
    def calc_pre(corr):
        precision = []
        for i in range(np.size(corr,0)):
            correct = 0
            for j in range(np.size(corr,1)):
                if corr[i,i] > corr[i,j]:
                    correct += 1
            precision.append(correct/(np.size(corr,1)-1))
        return precision
    # fusion of estimated rankings
    corr_fusion = delta*v_corr+(1-delta)*c_corr
    Rank_fusion,candidate_corr,candidate_ix = calc_rank(corr_fusion)
    Acc_fusion = calc_pre(corr_fusion)
    test_Rank_fusion = np.mean(Rank_fusion)
    test_Acc_fusion = np.mean(Acc_fusion)
    print('Average ranks from fusion results : {}'.format(test_Rank_fusion))
    print('Average accuracy from fusion results : {}'.format(test_Acc_fusion))
    return Rank_fusion,candidate_ix,test_Rank_fusion,test_Acc_fusion


# # Generate N-trial samples


V_pred = [0]*num_sub
C_pred = [0]*num_sub
for t in range(N_trial):
    X_sub,D_sub,N = multi_data()
    # MSBGM train
    Z_sub,SZZrep_sub,A_inv_sub,A0_inv_sub,gamma0_sub,gamma_sub,gamma_xx_sub,gamma_beta_sub,beta_inv_sub = initialize(X_sub,D_sub,Dy,num_sub)
    W_sub,WW_sub,beta_inv_sub,Z_sub = update(X_sub,D_sub,Dy,num_sub,maxiter_sub,Z_sub,SZZrep_sub,A_inv_sub,A0_inv_sub,gamma0_sub,gamma_sub,gamma_xx_sub,gamma_beta_sub,beta_inv_sub)
    # MVBGM train
    f_train,v_train,c_train,f_test = dataset(1)
    N,N_test,D = parameter(f_train,v_train,c_train,f_test)
    X,X_mean,X_norm = normalize(f_train,v_train,c_train)
    X_multi = [Z_sub,X[1],X[2]]
    D_multi = [Dy,D[1],D[2]]
    Z,SZZrep,A_inv,A0_inv,gamma0,gamma,gamma_xx,gamma_beta,beta_inv = initialize(X_multi,D_multi,Dz,num_multi)
    W,WW,beta_inv,_ = update(X_multi,D_multi,Dz,num_multi,maxiter,Z,SZZrep,A_inv,A0_inv,gamma0,gamma,gamma_xx,gamma_beta,beta_inv)
    
    for subject in range(5):
        f_train,v_train,c_train,f_test = dataset(subject+1)
        N,N_test,D = parameter(f_train,v_train,c_train,f_test)
        X,X_mean,X_norm = normalize(f_train,v_train,c_train)
        X_test = normalize_item(f_test,X_mean[0],X_norm[0])
        prZ_sub = predict_multi_sub(W_sub,WW_sub,beta_inv_sub,Dy,X_test,subject)
        v_pred,c_pred = predict(W,WW,beta_inv,Dz,prZ_sub)
        v_pred = renormalize_item(v_pred,X_mean[1],X_norm[1])
        c_pred = renormalize_item(c_pred,X_mean[2],X_norm[2])
        # sum of v_pred and c_pred
        V_pred[subject] += v_pred
        C_pred[subject] += c_pred

for subject in range(5):
    print('****************************************Estimation Result')
    print('subject:',subject+1)
    # average of N-trials
    V_pred_mean = V_pred[subject]/N_trial
    C_pred_mean = C_pred[subject]/N_trial
    Rank_fusion,candidate_ix,mean_rank,mean_acc = evaluate(V_pred_mean,C_pred_mean)


# # Estimated image categories for each test image


for test_index in range(1,51):
    # read image
    im = Image.open('../data/test_images/test{}.JPEG'.format(test_index))
    plt.imshow(im)
    plt.show()
    # read canididate
    f = open('../data/candidate_name.txt')
    candidate = f.readlines()
    print('test image category : {}'.format(candidate[test_index-1]))
    flag = 0
    for i in range(5):
        if i ==  Rank_fusion[test_index-1]-1:
            print(pycolor.RED + 'Rank {} : {}'.format(i+1,candidate[candidate_ix[test_index-1,i]]) + pycolor.END)
            flag = 1
        else:
            print('Rank {} : {}'.format(i+1,candidate[candidate_ix[test_index-1,i]]))
    if flag == 0:
        print('   *\n   *\n   *')
        print(pycolor.RED + 'Rank {} : {}'.format(Rank_fusion[test_index-1],candidate[test_index-1]) + pycolor.END)

