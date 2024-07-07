import numpy as np
from math import cos, sin, acos, asin, pi, sqrt
from numpy.linalg import inv, norm
from small_func import round_vec

#SGdata = np.load("data/type1sg_ops.npy", allow_pickle=True)
PG_list = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm',
            '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']

def get_rotation(R, return_list=False):
    det = np.linalg.det(R)
    assert np.allclose(abs(det), 1), det
    det = round(det)
    tmpR = det * R
    arg = (np.trace(tmpR) - 1) / 2
    if arg > 1:
        arg = 1
    elif arg < -1:
        arg = -1
    angle = acos(arg)
    axis = np.zeros(3)
    if abs(abs(angle) - pi) < 1e-4:
        for i in range(3):
            axis[i] = 1
            axis = axis + np.dot(tmpR, axis)
            if max(abs(axis)) > 1e-1:
                break
        assert max(abs(axis)) > 1e-1, 'can\'t find axis'
    elif abs(angle) > 1e-3:
        # standard case, see Altmann's book
        axis[0] = tmpR[2, 1] - tmpR[1, 2]
        axis[1] = tmpR[0, 2] - tmpR[2, 0]
        axis[2] = tmpR[1, 0] - tmpR[0, 1]
        axis = axis / sin(angle) / 2
    elif abs(angle) < 1e-4:
        axis[0] = 1
    # for non-orthogonal coordinates, axis may have norm>1, need to normalize
    axis = axis / np.linalg.norm(axis)
    axis = round_vec(axis)
    angle = round(angle / pi * 180)
    if return_list:
        return [det, angle, axis[0], axis[1], axis[2]]
    else:
        return angle, axis, det


def sort_rot(rot_list, tau_list):
    # sort rot according to det, angle, axis.
    nop = len(rot_list)
    det_angle_axis_list = []
    for rot in rot_list:
        angle, axis, det = get_rotation(rot)
        det_angle_axis_list.append([det, angle, axis[0], axis[1], axis[2]])

    det_angle_axis_list_orig = det_angle_axis_list.copy()
    det_angle_axis_list.sort(key=lambda x:(-x[0],x[1],x[2],x[3],x[4]))
    #print('sorted:',det_angle_axis_list)
    sorted_rot, sorted_tau, sort_order, sort_order_inv = [], [], np.zeros(nop, dtype=int), []
    for ith, item in enumerate(det_angle_axis_list):
        index = det_angle_axis_list_orig.index(item)
        sorted_rot.append(rot_list[index])
        sorted_tau.append(tau_list[index])
        sort_order[index] = ith  # i-th op in orig ops is sorted to sort_order[i]-th op
        sort_order_inv.append(index)  # i-th op in sorted ops is sort_order_inv[i]-th orig op

    return sorted_rot, sorted_tau, sort_order 


def int_mat(mat):
    # round the decimal part of the input mat
    roundmat = np.round(mat)
    assert np.linalg.norm(mat - roundmat) < 1e-6, (mat, roundmat)
    return roundmat

def PG_character_table():
    # generate a table for 32 PGs, shape=(32, 10)
    # column: nop, n_P, n_C2, n_M, n_C3, C_S3, n_C4, n_S4, n_C6, n_S6
    # row: 32 PGs
    PG_list = ['1', '-1', '2', 'm', '2/m', '222', 'mm2', 'mmm', '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm',
               '3', '-3', '32', '3m', '-3m', '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm', '23', 'm-3', '432', '-43m', 'm-3m']
    PG_chara_table = np.array([
    [1, 0, 0, 0,  0, 0, 0, 0, 0, 0],
    [2, 1, 0, 0,  0, 0, 0, 0, 0, 0],
    [2, 0, 1, 0,  0, 0, 0, 0, 0, 0],
    [2, 0, 0, 1,  0, 0, 0, 0, 0, 0],
    [4, 1, 1, 1,  0, 0, 0, 0, 0, 0],
    [4, 0, 3, 0,  0, 0, 0, 0, 0, 0],
    [4, 0, 1, 2,  0, 0, 0, 0, 0, 0],
    [8, 1, 3, 3,  0, 0, 0, 0, 0, 0],
    [4, 0, 1, 0,  0, 0, 2, 0, 0, 0],
    [4, 0, 1, 0,  0, 0, 0, 2, 0, 0],
    [8, 1, 1, 1,  0, 0, 2, 2, 0, 0],
    [8, 0, 5, 0,  0, 0, 2, 0, 0, 0],
    [8, 0, 1, 4,  0, 0, 2, 0, 0, 0],
    [8, 0, 3, 2,  0, 0, 0, 2, 0, 0],
    [16,1, 5, 5,  0, 0, 2, 2, 0, 0],
    [3, 0, 0, 0,  2, 0, 0, 0, 0, 0],
    [6, 1, 0, 0,  2, 2, 0, 0, 0, 0],
    [6, 0, 3, 0,  2, 0, 0, 0, 0, 0],
    [6, 0, 0, 3,  2, 0, 0, 0, 0, 0],
    [12,1, 3, 3,  2, 2, 0, 0, 0, 0],
    [6, 0, 1, 0,  2, 0, 0, 0, 2, 0],
    [6, 0, 0, 1,  2, 0, 0, 0, 0, 2],
    [12,1, 1, 1,  2, 2, 0, 0, 2, 2],
    [12,0, 7, 0,  2, 0, 0, 0, 2, 0],
    [12,0, 3, 4,  2, 0, 0, 0, 0, 2],
    [24,1, 7, 7,  2, 2, 0, 0, 2, 2],
    [12,0, 3, 0,  8, 0, 0, 0, 0, 0],
    [24,1, 3, 3,  8, 8, 0, 0, 0, 0],
    [24,0, 9, 0,  8, 0, 6, 0, 0, 0],
    [24,0, 3, 6,  8, 0, 0, 6, 0, 0],
    [48,1, 9, 9,  8, 8, 6, 6, 0, 0]])
    return PG_list, PG_chara_table

def rot_character(rot):
    # for a given rotation matrix (in cartesian coordinate), find its character table of length 10
    # i.e., [nop, n_P, n_C2, n_M, n_C3, C_S3, n_C4, n_S4, n_C6, n_S6]
    angle, axis, det = get_rotation(rot)
    if angle == 0 and det == 1:
        chara = [1, 0, 0, 0,  0, 0, 0, 0, 0, 0]
    elif angle == 0 and det == -1:
        chara = [1, 1, 0, 0,  0, 0, 0, 0, 0, 0]
    elif abs(angle) == 180 and det == 1:
        chara = [1, 0, 1, 0,  0, 0, 0, 0, 0, 0]
    elif abs(angle) == 180 and det == -1:
        chara = [1, 0, 0, 1,  0, 0, 0, 0, 0, 0]
    elif abs(angle) == 120 and det == 1:
        chara = [1, 0, 0, 0,  1, 0, 0, 0, 0, 0]
    elif abs(angle) == 120 and det == -1:
        chara = [1, 0, 0, 0,  0, 1, 0, 0, 0, 0]
    elif abs(angle) == 90 and det == 1:
        chara = [1, 0, 0, 0,  0, 0, 1, 0, 0, 0]
    elif abs(angle) == 90 and det == -1:
        chara = [1, 0, 0, 0,  0, 0, 0, 1, 0, 0]
    elif abs(angle) == 60 and det == 1:
        chara = [1, 0, 0, 0,  0, 0, 0, 0, 1, 0]
    elif abs(angle) == 60 and det == -1:
        chara = [1, 0, 0, 0,  0, 0, 0, 0, 0, 1]
    else:
        raise ValueError('Wrong rot!', rot, angle, axis, det)
    return np.array(chara)


def identify_PG_from_rot(rot_list, tol=1e-6):
    # For a given rot_list (in cartesian coordinate), identify its point group, and the main axis
    # return: PG_label: str, main_axis: str ('x/y/z')
    if rot_list == None:
        return 'Fail'
    if any([norm(np.imag(R)) > tol for R in rot_list]):
        # for rot with nonzero imag part, skip
        return 'Fail'
    else:
        rot_list = [np.real(R) for R in rot_list]

    PG_list, PG_chara_table = PG_character_table()
    rot_chara = np.array([rot_character(R) for R in rot_list])
    group_chara = np.sum(rot_chara, axis=0)
    PG_index_list = [index for index, row in enumerate(PG_chara_table) if norm(group_chara - row) == 0]
    if len(PG_index_list) == 0:
        print('Fail to identify PG!', group_chara)
        return 'Fail'
    else:
        PG_index = PG_index_list[0]

    PG_label = PG_list[PG_index]
    return PG_label

    

def PG_Bravais_latt(PG, main_axis='z'):
    # For a given PG label, return its compatible Bravais lattice name and primitive basis
   #if PG in ['1', '-1']:
   #   #Bravais_latt = ['P']
   #    # need to consider all Bravais lattice, because they give all possible supercell? FIXME 
   #    Bravais_latt = ['P', 'B', 'B2', 'B3', 'I', 'F', 'R']
   #elif PG in ['2', 'm', '2/m']:
   #    Bravais_latt = ['P', 'B', 'B2', 'B3', 'F']
   #elif PG in ['222', 'mm2', 'mmm']:
   #    Bravais_latt = ['P', 'B', 'B2', 'B3', 'I', 'F']
   #elif PG in ['4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm']:
   #    Bravais_latt = ['P', 'I']
   #elif PG in ['3', '-3', '32', '3m', '-3m']:
   #    Bravais_latt = ['P', 'R']
   #elif PG in ['6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm']:
   #    Bravais_latt = ['P']
   #elif PG in ['23', 'm-3', '432', '-43m', 'm-3m']:
   #    Bravais_latt = ['P', 'I', 'F']
   #else:
   #    raise ValueError('Wrong PG_label!', PG)
    Bravais_latt = ['P', 'B', 'B2', 'B3', 'I', 'F', 'R']
    return Bravais_latt

def find_PG_rot(PG_label):
    # find rot_list of a given PG. Some PG has two rot conventions.
    PG_corresponding_SG = [1, 2, 3, 6, 10, 16, 25, 47, 75, 81, 83, 89, 99, 111, 
            123, 143, 147, 149, 156, 162, 168, 174, 175, 177, 183, 187, 191, 195, 200, 207, 215, 221]
    SGdata = np.load("data/type1sg_ops.npy", allow_pickle=True)
    pg_rotC_list = []
    # the following PGs may have two conventions
    if PG_label == '-42m':
        pg_rotC_list.append(SGdata[111-1]['rotC'])
        pg_rotC_list.append(SGdata[115-1]['rotC'])
    elif PG_label == '32':
        pg_rotC_list.append(SGdata[149-1]['rotC'])
        pg_rotC_list.append(SGdata[150-1]['rotC'])
    elif PG_label == '3m':
        pg_rotC_list.append(SGdata[156-1]['rotC'])
        pg_rotC_list.append(SGdata[157-1]['rotC'])
    elif PG_label == '-3m':
        pg_rotC_list.append(SGdata[162-1]['rotC'])
        pg_rotC_list.append(SGdata[164-1]['rotC'])
    elif PG_label == '-6m2':
        pg_rotC_list.append(SGdata[187-1]['rotC'])
        pg_rotC_list.append(SGdata[189-1]['rotC'])
    else:
        ipg = PG_list.index(PG_label)
        pg_rotC_list.append(SGdata[PG_corresponding_SG[ipg] - 1]['rotC'])

    nop = len(pg_rotC_list[0]) // 2
    return [pg_rotC[0: nop] for pg_rotC in pg_rotC_list]
    
def identify_PG_from_gid(gid):
    # identify PG label for a given sg
    # return: PG_label
    SG_list = [[1], [2], [3,4,5], [6,7,8,9], np.arange(10,16), np.arange(16,25), np.arange(25,47), np.arange(47,75), 
               np.arange(75,81), [81,82], np.arange(83,89), np.arange(89,99), np.arange(99,111), np.arange(111,123), np.arange(123,143),
               np.arange(143,147), [147,148], np.arange(149,156), np.arange(156,162), np.arange(162,168), 
               np.arange(168,174), [174], [175,176], np.arange(177,183), np.arange(183,187), np.arange(187,191), np.arange(191,195),
               np.arange(195,200), np.arange(200,207), np.arange(207,215), np.arange(215,221), np.arange(221,231)]

    index = [i for i, lis in enumerate(SG_list) if gid in lis][0]
    return PG_list[index]


def find_rot_index(rot_list1, rot_list2, tol=1e-6):
    # find the index of rot of rot_list1 in rot_list2
    index_list = []
    for iR, R in enumerate(rot_list1):
        index = [i for i, op in enumerate(rot_list2) if norm(R - op) < tol][0]
        index_list.append(index)
    return np.array(index_list)



def find_SG_subPG(gid):
    # For a given SG, find all possible sub-PGs
    # return: a list of info, [{'sub_pg'：'XX', 't_index':n, 'rot_index':[....], 'transmat':[rot, tau]}, ...]
    # Method: enumerate Bilbao subsg data (k-index=1), 

    sg_data = SGdata[gid - 1]
    subsg_transmat_dict = subsg_transmat_data[gid - 1]
    nop = int(len(sg_data['rotC']) / 2)
   #print('SG rotC', sg_data['rotC'][:nop])
    print('\n\ngid: ', gid)

    gid_pg = identify_PG_from_gid(gid)
    subPG_list = [{'sub_pg': gid_pg, 't_index': 1, 'rot_index': np.arange(nop), 'transmat': [np.eye(3), np.zeros(3)]}]
    print(subPG_list[0])

    for isub, subsg in enumerate(subsg_transmat_dict['subsg']):
       #print('isub', isub, subsg)
        subsg_data = SGdata[subsg - 1] 
        nop_sub = int(len(subsg_data['rotC']) / 2)
        subsg_pg = identify_PG_from_gid(subsg)
        subsg_t_index = subsg_transmat_dict['t_index'][isub]

        for trans_rot, trans_tau in zip(subsg_transmat_dict['rot'][isub], subsg_transmat_dict['tau'][isub]):
           #print('transrot', trans_rot, trans_tau)
            # (R_sup, t_sup) = (R, t) @ (R_sub, t_sub) @ (R, t)^-1, where (R,t) is trans_rot/tau
            rot_transed = [trans_rot @ R @ inv(trans_rot) for R in subsg_data['rotC'][:nop_sub]]
           #rot_transed = [trans_rot @ R @ int_mat(inv(trans_rot)) for R in subsg_data['rotC'][:nop_sub]]
           #tau_transed = [trans_rot @ (-R_sub @ int_mat(inv(trans_rot)) @ trans_tau + tau_sub) + trans_tau) \
           #                    for R_sub, tau_sub in zip(subsg_data['rotC'][:nop_sub], subsg_data['tauC'][:nop_sub]]

            op_supsg_index = find_rot_index(rot_transed, sg_data['rotC'][:nop])

            if all([len(sub_dict['rot_index']) != len(op_supsg_index) or 
                    norm(sub_dict['rot_index'] - op_supsg_index) != 0 for sub_dict in subPG_list]):
                sub_dict = {'sub_pg': subsg_pg, 't_index': subsg_t_index, 'rot_index': op_supsg_index, 'transmat': [trans_rot, trans_tau]}
                subPG_list.append(sub_dict)
                print(sub_dict)

    return subPG_list

def save_SG_subPG_data():
    data = []
    for gid in range(1,231):
        subPG_list = find_SG_subPG(gid)
        data.append(subPG_list)
    np.save('data/SG_subPG_data.npy', data)


if __name__ == '__main__':
   #gid = 124
   #subPG_dict = find_SG_subPG(gid)
    save_SG_subPG_data()