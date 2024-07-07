from sys import argv, stdout
from os import path
from ctypes import CDLL, c_int, POINTER
from numpy.ctypeslib import ndpointer 
import numpy as np
# import sage.all as sage 

# ============================================================================= 
# 
# Load integer linear algebra library 
#
# =============================================================================
# _dll=CDLL(path.split(path.realpath(argv[0]))[0]+'/libintlng.so')
_dll=CDLL('/storage1/home/zysong/SSG/identifySSG/smith_form/libintlng.so')

_smith_form = _dll.smith_form_interface
_smith_form.argtypes = [c_int,c_int,POINTER(c_int),\
                        POINTER(c_int),POINTER(c_int),\
                        POINTER(c_int),POINTER(c_int)]
_smith_form.restype  = c_int

_row_echelon_form= _dll.row_echelon_form_interface
_row_echelon_form.argtypes = [ c_int, c_int, POINTER(c_int), \
                               POINTER(c_int), POINTER(c_int) ]
_row_echelon_form.restype = c_int

def smith_form_(M):
    n1,n2=M.shape
    if M.size>0:
        assert isinstance(M[0,0],np.int64), str(type(M[0,0]))
        #assert np.max(np.abs(M))<1e8, str(np.max(np.abs(M)))
    #
    M1=np.array(M.reshape(1,n1*n2),dtype=c_int)
    #
    L =np.zeros([n1*n1],dtype=c_int)
    LI=np.zeros([n1*n1],dtype=c_int)
    R =np.zeros([n2*n2],dtype=c_int)
    RI=np.zeros([n2*n2],dtype=c_int)
    #
    rank = _smith_form(c_int(n1),c_int(n2), M1.ctypes.data_as(POINTER(c_int)),\
            L.ctypes.data_as(POINTER(c_int)), LI.ctypes.data_as(POINTER(c_int)),\
            R.ctypes.data_as(POINTER(c_int)), RI.ctypes.data_as(POINTER(c_int)))
    #   
    return int(rank), np.array(M1.reshape(n1,n2),dtype='int'), \
       np.array(L.reshape(n1,n1),dtype='int'), np.array(LI.reshape(n1,n1),dtype='int'), \
       np.array(R.reshape(n2,n2),dtype='int'), np.array(RI.reshape(n2,n2),dtype='int')

def row_echelon_form(M):
    n1,n2=M.shape
    if M.size>0:
        assert isinstance(M[0,0],np.int64), str(type(M[0,0]))
        assert np.max(np.abs(M))<1e8, str(np.max(np.abs(M)))
    #
    M1=np.array(M.reshape(1,n1*n2), dtype=c_int)
    #
    L =np.zeros([n1*n1], dtype=c_int)
    LI=np.zeros([n1*n1], dtype=c_int)
    #
    rank = _row_echelon_form(c_int(n1), c_int(n2), M1.ctypes.data_as(POINTER(c_int)),\
            L.ctypes.data_as(POINTER(c_int)), LI.ctypes.data_as(POINTER(c_int)))
    return int(rank), np.array(M1.reshape(n1,n2),dtype='int'), \
                      np.array(L.reshape(n1,n1),dtype='int'), \
                      np.array(LI.reshape(n1,n1),dtype='int')

def col_echelon_form(M):
    rank, M1, RT, RIT = row_echelon_form(M.T)
    return rank, M1.T, RT.T, RIT.T



def smith_form(M):
    n1,n2 = M.shape
    if M.size==0:
        R  = np.eye(n2,dtype='int')
        RI = np.eye(n2,dtype='int')
        L  = np.eye(n1,dtype='int')
        LI = np.eye(n1,dtype='int')
        Lmd= np.zeros([n1,n2],dtype='int')
        rank=0
    else:
        if n1>1000 or n2>1000:
            suc = False
        else:
            rank, Lmd, L, LI, R, RI = smith_form_(M)
            suc = ( np.count_nonzero( np.dot(L,np.dot(M,R))-Lmd ) == 0 )
        #
        if not suc or np.max(np.abs(L))>10000:
            print('Warning: smith_form() fails in first try !')
            rank, M1, L_, LI_ = row_echelon_form(M) # M = L_^-1 @ M1
            rank, M2, R_, RI_ = col_echelon_form(M1) # M1 = M2 @ R_^-1
            rank, Lmd, L, LI, R, RI = smith_form_(M2) # M2 = L^-1 @ Lmd @ R^-1
            #
            L   = np.dot(L,L_)
            LI  = np.dot(LI_,LI)
            R   = np.dot(R_,R)
            RI  = np.dot(RI,RI_)
            #
            if np.count_nonzero( np.dot(L,np.dot(M,R))-Lmd ) == 0:
                print('Warning: smith_form() successes in second try !')
                stdout.flush()
            else:
                print('Warning: smith_form() fails in second try !')
                stdout.flush()
                assert False
            #
    return rank, Lmd, L, LI, R, RI


if __name__ == '__main__':
    M = np.eye(3)

