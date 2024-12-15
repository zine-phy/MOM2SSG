import numpy as np
from numpy.linalg import norm, inv, eigh, det
import spglib
from .small_func import *
from .PG_utils import sort_rot
import pickle
import os, sys

def identify_SG_lattice(gid):
    # identify Bravais lattice for a given sg, return: Brav_latt
    SGTricP = [1, 2]
    SGMonoP = [3, 4, 6, 7, 10, 11, 13, 14]
    SGMonoB = [5, 8, 9, 12, 15]
    SGOrthP = [16, 17, 18, 19, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
    SGOrthB1 = [20, 21, 35, 36, 37, 63, 64, 65, 66, 67, 68]
    SGOrthB2 = [38, 39, 40, 41]
    SGOrthF = [22, 42, 43, 69, 70]
    SGOrthI = [23, 24, 44, 45, 46, 71, 72, 73, 74]
    SGTetrP = [75, 76, 77, 78, 81, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94,
               95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 111, 112,
               113, 114, 115, 116, 117, 118, 123, 124, 125, 126, 127,
               128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138]
    SGTetrI = [79, 80, 82, 87, 88, 97, 98, 107, 108, 109, 110, 119, 120,
               121, 122, 139, 140, 141, 142]
    SGTrigP = [146, 148, 155, 160, 161, 166, 167]
    SGHexaP = [143, 144, 145, 147, 149, 150, 151, 152, 153, 154, 156,
               157, 158, 159, 162, 163, 164, 165, 168, 169, 170, 171,
               172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
               183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194]
    SGCubcP = [195, 198, 200, 201, 205, 207, 208, 212, 213, 215, 218,
               221, 222, 223, 224]
    SGCubcF = [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228]
    SGCubcI = [197, 199, 204, 206, 211, 214, 217, 220, 229, 230]

    # each row of prim_vec is a primitive base vector, and each column of prim_vec^-1 is a primitive base vec of BZ.
    if gid in SGTricP + SGMonoP + SGOrthP + SGTetrP + SGHexaP + SGCubcP:
        latt = 'P'
        prim_vec = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    elif gid in SGMonoB + SGOrthB1:
        latt = 'B'
        prim_vec = np.array([[1. / 2, -1. / 2, 0.], [1. / 2, 1. / 2, 0.], [0., 0., 1.]])
    elif gid in SGOrthB2:
        latt = 'B2'
        prim_vec = np.array([[1., 0., 0.], [0., 1. / 2, 1. / 2], [0., -1. / 2, 1. / 2]])
    elif gid in SGOrthI + SGTetrI + SGCubcI:
        latt = 'I'
        prim_vec = np.array([[-1. / 2, 1. / 2, 1. / 2], [1. / 2, -1. / 2, 1. / 2], [1. / 2, 1. / 2, -1. / 2]])
    elif gid in SGOrthF + SGCubcF:
        latt = 'F'
        prim_vec = np.array([[0., 1. / 2, 1. / 2], [1. / 2, 0., 1. / 2], [1. / 2, 1. / 2, 0.]])
    elif gid in SGTrigP:
        latt = 'R'
        prim_vec = np.array([[2. / 3, 1. / 3, 1. / 3], [-1. / 3, 1. / 3, 1. / 3], [-1. / 3, -2. / 3, 1. / 3]])
    else:
        raise ValueError('Wrong gid!', gid)
    # return transposed prim_vec, i.e., each col a prim vector
    return latt, prim_vec.T


def generate_coord(rot_list, tau_list):
    # generate coord list using {rot|tau}, starting from a given generic position
    gen_pts = [np.array([0.1722, 0.6933, 0.9344]),
                np.array([0.8399, 0.5677, 0.0655]),
                np.array([0.1234, 0.4567, 0.0876]),
                np.array([0.2468, 0.5721, 0.7834])]
    pos_list = []
    for R, t in zip(rot_list, tau_list):
        for p in gen_pts:
           #pos_list.append(latt_home(R @ p + t))
            pos_list.append(R @ p + t)

    return pos_list, len(gen_pts)


def identify_SG_from_cell(rotP_list, tauP_list, latt_prim, trig_hexag_latt=False):
    # generate a coordinate list and identify SG, return both gid and transformation matrix 
    def _C3C6_lattice(input='C3'):
        # for trigonal and hexagonal system, unit cell needs to change
        # In 14 Bravais latt, trigonal and hexagonal belong to the same latt, sharing the same basis vector
        # i.e., a1=a2, angle=2*pi/3
        return np.array([[1, 0, 0], [-0.5, np.sqrt(3)/2, 0], [0, 0, 1]])

   #gid, gid_label = identify_SG_from_symmetry(rotP_list, tauP_list)
   #if PG_label in ['3', '-3', '32', '3m', '-3m'] + ['6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm']:
   #if gid in range(143, 195):
    if trig_hexag_latt: 
        latt = latt_prim @ _C3C6_lattice(input='C3')
    else:
        latt = latt_prim

    pos_list, nspices = generate_coord(rotP_list, tauP_list)
    atom_num = np.arange(nspices).tolist() * len(rotP_list)
    cell = (latt, pos_list, atom_num)
    dataset = spglib.get_symmetry_dataset(cell)
    gid, gid_label, rot_std, tau_std, transmat, transtau, latt_std = dataset['number'], dataset['international'], \
                dataset['rotations'], dataset['translations'], dataset['transformation_matrix'], dataset['origin_shift'], dataset['std_lattice']
   #print(gid, gid_label, rot_std, transmat, latt_std)
    
    if len(rot_std) != len(rotP_list):
        print('identified gid has fewer operations!', len(rot_std), len(rotP_list))
       #prim_cell = spglib.find_primitive(cell, symprec=1e-5)
       #print('original cell:', cell)
       #print('spglib found primitive cell:', prim_cell)
       #dataset = spglib.get_symmetry_dataset(cell)
       #print('primitive cell gid:', dataset['number'])
        return None, None
    return gid, gid_label


def identify_SG_from_symmetry(rot_list, tau_list):
    # for input symmetry operations, identify space group using spglib
    hall_num = spglib.get_hall_number_from_symmetry(rot_list, tau_list, symprec=1e-5)
    sg_info = spglib.get_spacegroup_type(hall_num)
    if sg_info == None:
        print('spglib Fail! Return None!')
        return None, None

    gid, gid_label = sg_info['number'], sg_info['international_short']
    sym_spglib = spglib.get_symmetry_from_database(hall_num)
    # symmetries from spglib may have multiple identity with different lattice translations for different bravais lattice
    n_multiple = sum([np.allclose(R, np.eye(3)) for R in sym_spglib['rotations']])
   #assert len(sym_spglib['rotations']) // n_multiple == len(rot_list), (sym_spglib, rot_list, tau_list)
    if len(sym_spglib['rotations']) // n_multiple != len(rot_list):
        return None, None

    return gid, gid_label



def test():
   #rot_list = [np.eye(3), np.diag((-1,-1,1)), np.diag((-1,1,-1)), np.diag((1,-1,-1))]
   #tau_list = [np.array([0,0,0]), np.array([0,0,1/2]), np.array([0,0,1/2]), np.array([0,0,0])]
    rot_list = [np.eye(3), np.diag((1,-1,1))]
    tau_list = [np.array([0,0,0]), np.array([0,1/2,0])]
    print('original ops:', rot_list, tau_list)
    latt = 'B'
    prim_basis = find_Bravais_prim_vec(latt)

    rotP_list = [inv(prim_basis.T) @ R @ prim_basis.T for R in rot_list]
    tauP_list = [inv(prim_basis.T) @ t for t in tau_list]
    print('prim:', rotP_list, tauP_list, prim_basis)
    gid, gid_label = identify_SG_from_symmetry(rotP_list, tauP_list)
    print('gid', gid, gid_label)

    prim_trans = find_prim_latt_trans_spglib(latt)
    rot_list_conv = [R for R in rot_list]
    tau_list_conv = [t for t in tau_list]
    for ptau in prim_trans:
        rot_list_conv.extend([R for R in rot_list])
        tau_list_conv.extend([latt_home(t + ptau) for t in tau_list])
    print('conv:', prim_trans, rot_list_conv, tau_list_conv)
    hall_num = spglib.get_hall_number_from_symmetry(rot_list_conv, tau_list_conv, symprec=1e-5)
    sg_info = spglib.get_spacegroup_type(hall_num)
    gid = sg_info['number']
    sym_spglib = spglib.get_symmetry_from_database(hall_num)
    print('gid', gid, sym_spglib)
    print('hall_num', hall_num)

def test_hall():
   #hall_num = 9 # C2, + [0.5, 0.5, 0. ]
   #hall_num = 11 # I2, + [0.5, 0.5, 0.5]
   #hall_num = 122 # F222, sg22, + [0. , 0.5, 0.5], [0.5, 0. , 0.5], [0.5, 0.5, 0. ]
    hall_num = 433 # R3, sg146, + [2/3, 1/3, 1/3], [1/3, 2/3, 2/3] 
    sym_spglib = spglib.get_symmetry_from_database(hall_num)
    print('hall_num', hall_num, sym_spglib)

def test_sym():
    rotP_list = [np.eye(3), np.array([[0, -1, 0],[-1, 0, 0],[0, 0, -1]])]
    tauP_list = [np.zeros(3), np.zeros(3)]
    gid, gid_label = identify_SG_from_symmetry(rotP_list, tauP_list)
    print('gid', gid, gid_label)

def test1():
    rot_list = [np.eye(3), np.diag((-1,1,-1))]
    rotP_list = [np.eye(3), np.array([[-1,-3,0],[0,1,0],[0,1,-1]])]
    tau_list = [np.zeros(3), np.zeros(3)]
    prim_latt = np.array([[1,0,1],[1.5,1.5,0],[0,0,3]])
    identify_SG_from_cell(rotP_list, tau_list, prim_latt)



