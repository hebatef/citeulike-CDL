import numpy as np
from pandas import read_csv

def read_rating(file_path, has_header=True):
    rating_mat = list()
    with open(file_path) as fp:
        if has_header is True:
            fp.readline()
        for line in fp:
            line = line.split(',')
            user, item, rating = line[0], line[1], line[2]
            rating_mat.append( [user, item, rating] )
    return np.array(rating_mat).astype('float32')

def read_feature(file_path):
    feat_mat = read_csv(file_path, sep=',')
    assert( np.all(feat_mat['id'] == feat_mat.index) )
    return feat_mat.drop('id', 1).as_matrix()

def get_mult():
    X = read_mult('data/dummy/mult.dat',8000).astype(np.float32)
    return X

def read_mult(f_in='data/dummy/mult.dat',D=8000):
    fp = open(f_in)
    lines = fp.readlines()
    X = np.zeros((len(lines),D))
    for i,line in enumerate(lines):
        strs = line.strip().split(' ')[1:]
        for strr in strs:
            segs = strr.split(':')
            X[i,int(segs[0])] = float(segs[1])
    arr_max = np.amax(X,axis=1)
    X = (X.T/arr_max).T
    return X

def read_user(f_in='data/cf-train-1-users.dat',num_u=5551,num_v=16980):
    fp = open(f_in)
    R = np.mat(np.zeros((num_u,num_v), dtype=np.int32))
    for i,line in enumerate(fp):
        segs = line.strip().split(' ')[1:]
        for seg in segs:
            R[i,int(seg)] = 1
    return R
