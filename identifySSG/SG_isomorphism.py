import numpy as np
from numpy.linalg import norm, inv, eigh, det
import spglib
import pickle
import os, sys
from itertools import permutations
import sympy as sp
from small_func import *
from PG_utils import sort_rot, identify_PG_from_gid
from SG_utils import identify_SG_lattice
# from sim_trans import find_similarity_transformation, rep_similarity_transformation
from time import time
path = '/storage1/home/zysong/SSG/SpinSG_final/SSG_codes'
sys.path.append(path)
SG_data = np.load("%s/data/type1sg_ops.npy"%path, allow_pickle=True)
SG_subPG_data = np.load("%s/data/SG_subPG_data.npy"%path, allow_pickle=True)


def get_sg_iso_tau(rot_list, tau_list, iso_R, tol=1e-6):
    '''
    For symorphic SG and W=(A, 0), check if A @ G @ A^-1 = G, where the ordering of ops may change, 
    but the matrix form of R cannot change.
    For non-symorphic SG, check first if the point group part satisfies A @ P_G @ A^-1 = P_G,
    and then solve if there exist t, s.t W=(A, t) satisfies W @ G @ W^-1 = G
    
    return: is_iso: True/False, iso_tau, and 
            iso_mapping=[j1, j2, ..., jn]: f(G) = W G W^-1, map nth op to jth op (for sorted G ops)
            (rot_trans_list[i] = rot_orig_list[iso_mapping[i]])

    1. compare G and G': first sort ops using (angle, axis, det), then compare sorted op directly
    2. input rot_list and tau_list are sorted
    3. input ops are in G prim latt, iso_R also in G prim latt
    '''
    is_iso, iso_tau, iso_mapping = False, None, []
    nop = len(rot_list)
    rot_trans_list = [round_mat(iso_R @ R @ inv(iso_R)) for R in rot_list]
    tau_trans_list = [iso_R @ t for t in tau_list]

    sorted_rot_trans_list, sorted_tau_trans_list, sort_order = sort_rot(rot_trans_list, tau_trans_list)
    if all([np.allclose(R1, R2) for R1, R2 in zip(rot_list, sorted_rot_trans_list)]):
        iso_mapping = [find_mat(R_trans, rot_list) for R_trans in rot_trans_list]

        # if transformed tau are identical, then iso_tau is zero
        if all([norm(t1 - t2) < tol for t1, t2 in zip(tau_list, sorted_tau_trans_list)]):
            is_iso, iso_tau = True, [np.zeros(3)]
        else:
            # compute iso_tau using sorted ops, (iso_R, t) @ (R1, t1) = (R2, t2) @ (iso_R, t)
            # ==> (R2 - 1) t = iso_R @ t1 - t2 % 1, transform into M @ t = w % [1]
            M = np.zeros((3 * nop, 3))
            w = np.zeros(3 * nop)
            for iop, R1, t1, iop2, R2 in zip(range(nop), rot_list, tau_list, iso_mapping, rot_trans_list):
                t2 = tau_list[iop2]
                M[iop * 3: (iop + 1) * 3, :] = R2 - np.eye(3)
                w[iop * 3: (iop + 1) * 3] = iso_R @ t1 - t2

            # solve M @ t = w % [1]
            trial_tlist = solve_linear_inhomogeneous(M, w, mod_num=[1,1,1], allow_infinite_sol=True)
            if len(trial_tlist) > 0:
                is_iso = True
                iso_tau = trial_tlist
                
                # check result
                # tau_trans_list = [iso_tau + iso_R @ t - iso_R @ R @ inv(iso_R) @ iso_tau
                #                     for R, t in zip(rot_list, tau_list)]
                # assert all([np.allclose(rot_trans_list[i], rot_list[iso_mapping[i]]) for i in range(nop)]), (
                #             rot_list, rot_trans_list, iso_mapping)
                # assert all([norm(np.mod(round_vec(tau_trans_list[i] - tau_list[iso_mapping[i]]),
                #                         [1,1,1])) < tol for i in range(nop)]), (tau_trans_list, tau_list)
                
    return is_iso, iso_tau, iso_mapping


def find_sg_iso_transform(gid, tol=1e-6, test=0):
    '''
    find (A, t) G (A, t)^-1 = G (the ordering of ops may change, need to find group isomorphism)
    Method:
        1. For a given crystal system, the isomorphism of rotation part is easy to obtain (given manually) 
        2. For each sg, first check the rotation isomorphism, then solve translation part.
    Output:
        a dictionary {'iso_rot': [R, ...], 'iso_tau': [t, ...]}
    '''
    gid_data = SG_data[gid - 1]
    nop = len(gid_data['rotP']) // 2
    sorted_rot_list, sorted_tau_list, sort_order = sort_rot(gid_data['rotP'][0: nop], gid_data['tauP'][0: nop])
    prim_basis = identify_SG_lattice(gid)[1] # each col is a prim basis vector

    if gid in [1, 2]:
        # any SL(3, Z) is an isomorphism
        iso_sg = 207 
    elif gid in range(3, 16):
        # any SL(2, Z) along x, z axis is an isomorphism, and C2x, C2z
        # here use 207 to include C2x, C2z (extra ops will be abandoned)
        iso_sg = 207 
    elif gid in range(16, 75):
        # PG: 432 (sg 207)
        iso_sg = 207
    elif gid in range(75, 143):
        # PG: 422 (sg 89)
        iso_sg = 89
    elif gid in range(143, 168):
        # PG: 32 (sg 149)
        iso_sg = 149
    elif gid in range(168, 195):
        # PG: 622 (sg 177)
        iso_sg = 177
    elif gid in range(195, 231):
        # PG: 432 (sg 207)
        iso_sg = 207

    iso_trans = {'iso_rot': [], 'iso_tau': [], 'iso_mapping': []}
    iso_rot_candidates = SG_data[iso_sg - 1]['rotC']
    nop_iso = len(iso_rot_candidates) // 2
    iso_rot_candidates = iso_rot_candidates[0: nop_iso]

    if gid in [1, 2]:
        # triclinic system
        iso_rot_candidates.extend(find_triclinic_trial_iso())

    if gid in range(3, 16):
       #if not monoclinic_simplify: 
            # When finding eqv 3D reps, does not need to consider trial iso, and monoclinic_simplify=True
            # otherwise monoclinic_simplify=False and consider trial isomorphism with bound [-Max, Max+1)
        iso_rot_candidates.extend(find_monoclinic_trial_iso(Max=2)) # 50
       #iso_rot_candidates.extend(find_monoclinic_trial_iso(Max=12)) # 50
       #iso_rot_candidates.extend(find_monoclinic_trial_iso(Max=36)) # 50
       #iso_rot_candidates.extend(find_monoclinic_trial_iso(Max=50)) # 50

    print('conv candidates:', iso_rot_candidates) if test else None
    if norm(prim_basis - np.eye(3)) > tol:
        # transform iso_R to G prim basis, and keep only integer matrices
        iso_rot_candidates = [round_mat(inv(prim_basis) @ R @ prim_basis) for R in iso_rot_candidates]
        iso_rot_candidates = [R for R in iso_rot_candidates if vec_is_int(R)]
    print('num of iso_rot_candidates:', len(iso_rot_candidates)) if test else None
    print('candidates', iso_rot_candidates) if test else None

    for ith, iso_R in enumerate(iso_rot_candidates):        
        print('\n', ith, iso_R.tolist()) if test else None
        is_iso, iso_tau, iso_mapping = get_sg_iso_tau(sorted_rot_list, sorted_tau_list, iso_R)
        if is_iso:
            iso_R = prim_basis @ iso_R @ inv(prim_basis)
            conv_iso_tau = []
            for t in iso_tau:
                conv_iso_tau.append(prim_basis @ t)
            iso_trans['iso_rot'].append(iso_R)  # iso_rot in G prim latt
            iso_trans['iso_tau'].append(conv_iso_tau)  # iso_tau in G prim latt
            iso_trans['iso_mapping'].append(iso_mapping)
            print('find isomorphism:\n', iso_R, iso_tau) if test else None
        else:
            print('Not isomorphism!') if test else None

    print('total sol:', len(iso_trans['iso_rot'])) if test else None
    return iso_trans


def check_eqv_subsg(H1, H2, gid_data, S, sg_iso_trans, tol=1e-6, test=0):
    '''
    For two input subsg of the same supercell S of SG G, check if they are equivalent
    If H1 = H2 = H, check if H is invariant under iso_trans
    
    Method: two subsg H1, H2 are equivalent if there exists an sg isomorphism W=(iso_R, iso_tau), 
    s.t. W @ H1 @ W^-1 = H2, and iso_R @ S @ C = S
    
    Note that translation part of W may not be unique, and needs to be solved
    by combining W @ G = G @ W and W @ H1 = H2 @ W
    
    The iso_R must be integer matrix in H prim latt, i.e., inv(S) @ iso_R @ S is int matrix
    '''
    nop_G = len(gid_data['rotC']) // 2
    G_rot_list, G_tau_list, sort_order = sort_rot(gid_data['rotP'][0: nop_G], gid_data['tauP'][0: nop_G])

    # sort H1, H2 using sort_order of G
    H1_idx_in_sorted_G = [sort_order[i] for i in H1['rot_index']]
    H1_rotP = [round_mat(inv(S) @ G_rot_list[i] @ S) for i in H1_idx_in_sorted_G] # in H prim latt, use supercelll S, inv(S) @ R @ S, from G prim to H prim
    H1_tauP = H1['tau_prim']

    H2_idx_in_sorted_G = [sort_order[i] for i in H2['rot_index']]
    H2_rotP = [round_mat(inv(S) @ G_rot_list[i] @ S) for i in H2_idx_in_sorted_G]
    H2_tauP = H2['tau_prim']
    assert all([vec_is_int(R) for R in H1_rotP + H2_rotP]), (H1_rotP, H2_rotP)
    nop_H = len(H1_rotP)
   #print('G_rotP, tauP:', G_rot_list, G_tau_list) if test else None
   #print('H1_rotP, tauP:', H1_rotP, H1_tauP) if test else None
   #print('H2_rotP, tauP:', H2_rotP, H2_tauP) if test else None

    # use sympy to generate symbolic equation Mt=w
   #a, b, c, d = sp.symbols('a b c d')
   #iso_R_sp = sp.Matrix([[a, 0, b], [0, 1, 0], [c, 0, d]])
   #Prim = sp.Matrix([[sp.Rational(1/2),sp.Rational(1/2),0],[-sp.Rational(1/2),sp.Rational(1/2),0],[0,0,1]])
   #iso_R_sp_P = Prim.inv() @ iso_R_sp @ Prim
   #iso_R_sp_PS = sp.Matrix(S).inv() @ iso_R_sp_P @ sp.Matrix(S)
    
    for ith, iso_R in enumerate(sg_iso_trans['iso_rot']):
        # first check whether rotation part is matched
        iso_mapping = sg_iso_trans['iso_mapping'][ith]
        H1_idx_trans = [iso_mapping[i] for i in H1_idx_in_sorted_G]
        if np.allclose(np.sort(H1_idx_trans), np.sort(H2_idx_in_sorted_G)):
            iso_mapping_H = [H2_idx_in_sorted_G.index(i) for i in H1_idx_trans]
            iso_R_Hprim = round_mat(inv(S) @ iso_R @ S)
            assert vec_is_int(iso_R_Hprim), iso_R_Hprim

            # if W = (iso_R, iso_tau) transform the rotations of H1 to that of H2, solve iso_tau
            # transform W G = G W and W H1 = H2 W into M @ t = w % [1]
            # ==> (R2 - 1) S t = iso_R @ t1 - t2, for op in G, written in G prim latt, t in H prim latt
            #     (R2 - 1) t = S^-1 @ iso_R @ S @ t1 - t2, for op in H1, H2,
            # written in H prim latt, where S is supercell, S^-1 @ iso_R @ S is iso_R in H prim latt  
            # solution t is in H prim latt
            M1 = np.zeros((3 * nop_G, 3))
            w1 = np.zeros(3 * nop_G)
           #w_sp = sp.zeros(3 * nop_G + 3 * nop_H, 1)
            for iop, R1, t1, iop2 in zip(range(nop_G), G_rot_list, G_tau_list, iso_mapping):
                R2 = G_rot_list[iop2]
                t2 = G_tau_list[iop2]
                M1[iop * 3: (iop + 1) * 3, :] = (R2 - np.eye(3)) @ S
                w1[iop * 3: (iop + 1) * 3] = iso_R @ t1 - t2
               #for icol in range(3):
               #    w_sp[iop * 3 + icol] = (iso_R_sp_P @ numpy_array_to_sympy(t1) - numpy_array_to_sympy(t2))[icol]

            M2 = np.zeros((3 * nop_H, 3))
            w2 = np.zeros(3 * nop_H)
            for iop, R1, t1 in zip(range(nop_H), H1_rotP, H1_tauP):
                iop2 = iso_mapping_H[iop]
                R2, t2 = H2_rotP[iop2], H2_tauP[iop2]
                M2[iop * 3: (iop + 1) * 3, :] = (R2 - np.eye(3))
                w2[iop * 3: (iop + 1) * 3] = iso_R_Hprim @ t1 - t2 
               #for icol in range(3):
               #    w_sp[3 * nop_G + iop * 3 + icol] = (iso_R_sp_PS @ numpy_array_to_sympy(t1) - numpy_array_to_sympy(t2))[icol]

            M = np.vstack((M1, M2))
            w = np.hstack((w1, w2))
            print('iso_R', iso_R, iso_R_Hprim) if test else None
            print('M1, w1', M1, w1) if test else None
            print('M2, w2', M2, w2) if test else None
           #print('w_sp', w_sp) if test else None
           #print('iso_R in S', iso_R_sp_PS) if test else None
           #W0 = Prim @ iso_R @ Prim.inv()
           #w_sp2np = np.array(w_sp.replace(a, W0[0,0]).replace(b, W0[0,2]).replace(c, W0[2,0]).replace(d, W0[2,2])).astype(float)
           #assert norm(w.reshape((3*(nop_G+nop_H), 1)) - w_sp2np) < 1e-6, (w_sp2np, W0) # have been checked, but do not apply to C2x, C2z
           #show_linear_inhomogeneous_sp(M, w_sp)

            # solve M @ t = w % [1]
            trial_tlist = solve_linear_inhomogeneous(M, w, mod_num=[1,1,1], allow_infinite_sol=1)

            if len(trial_tlist) > 0:
                iso_tau_Gprim_list = [S @ iso_tau_Hprim for iso_tau_Hprim in trial_tlist]
               #for iso_tau_Gprim, iso_tau_Hprim in zip(iso_tau_Gprim_list, trial_tlist):
               #    # check result of W G = G W
               #    G_rot_trans_list = [round_mat(iso_R @ R @ inv(iso_R)) for R in G_rot_list] 
               #    G_tau_trans_list = [iso_tau_Gprim + iso_R @ t - iso_R @ R @ inv(iso_R) @ iso_tau_Gprim
               #                        for R, t in zip(G_rot_list, G_tau_list)]
               #    assert all([np.allclose(G_rot_trans_list[i], G_rot_list[iso_mapping[i]]) for i in range(nop_G)]), (
               #                G_rot_list, G_rot_trans_list, iso_mapping)
               #    assert all([norm(np.mod(round_vec(G_tau_trans_list[i] - G_tau_list[iso_mapping[i]]),
               #                [1,1,1])) < tol for i in range(nop_G)]), (G_tau_trans_list, G_tau_list, iso_mapping)
               #    # check result of W H1 = H2 W
               #    H1_rot_trans_list = [round_mat(iso_R_Hprim @ R @ inv(iso_R_Hprim)) for R in H1_rotP]
               #    H1_tau_trans_list = [iso_tau_Hprim + iso_R_Hprim @ t - iso_R_Hprim @ R @ inv(iso_R_Hprim) @ iso_tau_Hprim
               #                        for R, t in zip(H1_rotP, H1_tauP)]
               #    assert all([np.allclose(H1_rot_trans_list[i], H2_rotP[iso_mapping_H[i]]) for i in range(nop_H)]), (
               #                H1_rot_trans_list, H2_rotP, iso_mapping_H, 
               #                iso_R, iso_R_Hprim, H1_idx_in_sorted_G, H2_idx_in_sorted_G, H1_idx_trans, G_rot_list)
               #    assert all([norm(np.mod(round_vec(H1_tau_trans_list[i] - H2_tauP[iso_mapping_H[i]]),
               #                [1,1,1])) < tol for i in range(nop_H)]), (H1_tau_trans_list, H2_tauP, iso_mapping_H)

                return True, iso_tau_Gprim_list
    return False, None


def find_sg_uneqv_subsg(gid, supercell, subsg_data, tol=1e-6, test=False):
    '''
    For a given sg and its supercell A, find unequivalent subPG
    
    Definition of equivalent subsg under a given supercell S: 
    two subsg H1, H2 are equivalent if there exists an sg isomorphism W, 
    s.t. W @ H1 @ W^-1 = H2, and W @ S @ C = S -> S^-1 @ W^-1 @ S is integer matrix
    
    Method:
    1. first enumerate SG isomorphisms {W} that is compatible with supercell S
    2. for subsg with the same PG (main axis can be different), find uneqv subsg
    '''
    # load sg data
    gid_data = SG_data[gid - 1]
    nop = len(gid_data['rotP']) // 2
    sorted_rot_list, sorted_tau_list, sort_order = sort_rot(gid_data['rotP'][0: nop], gid_data['tauP'][0:nop])
    prim_basis = identify_SG_lattice(gid)[1] # each col is a prim basis vector
    
    # first check if all subsg has different sg num, if so, they are all uneqv, and we do not need to use sg isomorphisms
    subSG_nums = set([d['subsg'] for d in subsg_data])
    num_subsg_of_each_SG = np.zeros(len(subSG_nums), dtype=int)
    for isg, sg_num in enumerate(subSG_nums):
        num_subsg_of_each_SG[isg] = len([d for d in subsg_data if d['subsg'] == sg_num])
    if all([n <= 1 for n in num_subsg_of_each_SG]):
        print('No or only one subsg has sg num=', subSG_nums, '  Total subsg num=', len(subsg_data), 'of supercell=', supercell.tolist())
        return subsg_data
        
    # step 1: find sg isomorphism compatible with supercell S => S^-1 @ W @ S is an int matrix
    sg_iso_dict = find_sg_iso_transform(gid)
    sg_iso_Sinv = {'iso_rot': [], 'iso_mapping': []}
    for ith, iso_R in enumerate(sg_iso_dict['iso_rot']):
        if vec_is_int(inv(supercell) @ iso_R @ supercell):
            sg_iso_Sinv['iso_rot'].append(iso_R)
            sg_iso_Sinv['iso_mapping'].append(sg_iso_dict['iso_mapping'][ith])

   ## step 2: find uneqv subsg with the same PG
   ## collect subPG labels
   #subPG_data = SG_subPG_data[gid - 1]
   #subPG_labels = []
   #for subPG_dict in subPG_data:
   #    if subPG_dict['sub_pg'] not in subPG_labels:
   #        subPG_labels.append(subPG_dict['sub_pg'])

   ## find uneqv subsg with the same PG, 
   ## Note: consider subsg with same PG is complete. Maybe only need to consider same SG, which is faster.
   #uneqv_subsg_data = []
   #for ipg, pg_label in enumerate(subPG_labels):
   #    subsg_of_pg = [d for d in subsg_data if identify_PG_from_gid(d['subsg']) == pg_label]
   #    uneqv_subsg_of_pg = []
   #    for ith, subsg_info in enumerate(subsg_of_pg):
   #        if (len(uneqv_subsg_of_pg) == 0 or
   #                all([not check_eqv_subsg(subsg_info, d, gid_data, supercell, sg_iso_Sinv)[0] 
   #                    for d in uneqv_subsg_of_pg])):
   #            uneqv_subsg_of_pg.append(subsg_info)
   #            uneqv_subsg_data.append(subsg_info)

    # step 2: find uneqv subsg of the same SG 
    uneqv_subsg_data = []
    for isg, sg_num in enumerate(subSG_nums):
        subsg_of_SG = [d for d in subsg_data if d['subsg'] == sg_num]
        if len(subsg_of_SG) > 1:
            if test:
                print('\n\n=============================') if test else None
                print('Num of subsg of the same SG > 1, subsg=%d, num subsg=%d, supercell= '%(sg_num, len(subsg_of_SG)), supercell.tolist())
                for d in subsg_of_SG:
                    print(d['tau_prim'].tolist())
                print('=============================\n\n')
            uneqv_subsg_of_SG = []
            for ith, subsg_info in enumerate(subsg_of_SG):
                if (len(uneqv_subsg_of_SG) == 0 or
                    all([not check_eqv_subsg(subsg_info, d, gid_data, supercell, sg_iso_Sinv)[0] 
                        for d in uneqv_subsg_of_SG])):
                    uneqv_subsg_of_SG.append(subsg_info)
                    uneqv_subsg_data.append(subsg_info)

            if len(subsg_of_SG) != len(uneqv_subsg_of_SG):
                print('Find uneqv subsg!') if test else None
       #else:
       #    print('No or only one subsg of SG=%d'%sg_num, len(subsg_of_SG), len(subsg_data), supercell.tolist())
                
    return uneqv_subsg_data


def find_quot_mapping(quot_group_data, iso_R, iso_t):
    '''
    Find iso_mapping_Q for quot group ops
    '''
    quot_rot, quot_tau = quot_group_data.rotC_fulllist, quot_group_data.tauC_fulllist
    H_latt_PinC = quot_group_data.H_latt_PinC
    
    iso_mapping_Q = []
    for R, t in zip(quot_rot, quot_tau):
        # (R_trans, t_trans) = (iso_R, iso_t)^-1 (R, t) (iso_R, iso_t)
        R_trans, t_trans = op_prod(inv(iso_R), -inv(iso_R) @ iso_t, *op_prod(R, t, iso_R, iso_t))
       #R_trans = round_mat(inv(iso_R) @ R @ iso_R), inv(iso_R) @ t
       #t_trans = -inv(iso_R) @ (iso_t - t + R @ iso_t)
        idx = quot_group_data._find_op_in_quot_group(R_trans, t_trans)
        iso_mapping_Q.append(idx)

    return iso_mapping_Q


def check_rep_eqv_using_ortho_thm(D1_chara, D2_mats, num_ir_D2):
    '''
    For two input (reducible) rep D1, D2, split D2 into IRREPs (ir1, ...), and compute orthogonal relation between (D1, ir_i).
    If (D1, ir_i) = 1 for each ir_i, then D1, D2 are equivalent
    Note: not necessary, compare characters directly is enough.
    '''
    nop = len(D1_chara)
    D2_split_chara = []
    if num_ir_D2 == 1: # 3D rep
        D2_split_chara.append([np.trace(M) for M in D2_mats])
    elif num_ir_D2 == 2: # 1+2D rep
        D2_split_chara.append([M[0, 0] for M in D2_mats])
        D2_split_chara.append([M[1, 1] + M[2, 2] for M in D2_mats])
        assert all([np.allclose(M, mat_direct_sum(M[0:1, 0:1], M[1:3, 1:3])) for M in D2_mats]), ('D2 not 1+2D!', D2_mats)
    elif num_ir_D2 == 3: # 1+1+1D rep
        D2_split_chara.append([M[0, 0] for M in D2_mats])
        D2_split_chara.append([M[1, 1] for M in D2_mats])
        D2_split_chara.append([M[2, 2] for M in D2_mats])
        assert all([np.allclose(M, np.diag((M[0,0], M[1,1], M[2,2]))) for M in D2_mats]), ('D2 not 1+1+1D!', D2_mats)       
    
    found_uneqv = False 
    for ith, ir_chara in enumerate(D2_split_chara):
        dot = sum([ np.conjugate(c1) * c2 for c1, c2 in zip(D1_chara, ir_chara) ]) / nop
       #assert np.allclose(dot, np.round(dot.real)), ('float nir!', dot, D1_chara, ir_chara, ith)
       #nir = np.round(dot.real)
        if not np.allclose(dot, 1):
            found_uneqv = True

    if found_uneqv:
        return False
    else:
        return True


def check_eqv_3Drep(D1, D2, sg_iso_Hinv, iso_mapping_Q2PG, tol=1e-6, test=False):
    '''
    For two input SSGs (Q|D1), (Q|D2), check whether they are equivalent.
    D1 and D2 must eqv to the same PG.
    
    Method: check if there exists an sg isomorphism W=(iso_R, iso_tau), 
    s.t. W @ (Q|D1) = (Q|D2) @ W
    
    Use iso_mapping_Q to check if it maps D1 to D2
    '''
    def _permute_irrep(D):
        # For input 3D rep, which can be 1+1+1D, 1+2D, or 3D.
        def _perm111(M, perm):
            diag = M.diagonal()
            M_perm = np.diag((diag[perm[0]], diag[perm[1]], diag[perm[2]]))
            return M_perm
        def _perm12(M):
            M_perm = np.zeros((3,3), dtype=complex)
            M_perm[0:2, 0:2] = M[1:3, 1:3]
            M_perm[2,2] = M[0,0]
            return M_perm
        # 1+1+1D rep, consider all 3rd order permutations
        if sum([s == '_' for s in D.label]) == 2: 
            perm_list = permutations([0, 1, 2], 3)
            perm_rep_mat = []
            for perm in perm_list:
                perm_rep_mat.append([_perm111(m, perm) for m in D.matrices])
        # 1+2D rep, consider 2nd order permutations, i.e., (0, 1) (1, 0)
        # In this case, D must be 1+2D (not 2+1D)
        elif sum([s == '_' for s in D.label]) == 1:
            perm_rep_mat = [[m for m in D.matrices]]
            perm_rep_mat.append([_perm12(m) for m in D.matrices])
        # 3D rep, no permutation
        else:
            perm_rep_mat = [[m for m in D.matrices]]
        return perm_rep_mat
    
    if D1.eqv_pg_label != D2.eqv_pg_label:
        # if D1 and D2 have different eqv pg, they must be uneqv
        return False 
    
    # two reps are equivalent if their characters are the same
    for ith, iso_mapping_Q in enumerate(sg_iso_Hinv['iso_mapping_Q']):
        D1_chara = [D1.characters[val] for val in iso_mapping_Q2PG.values()]
        D2_chara = [D2.characters[val] for val in iso_mapping_Q2PG.values()]
        if all([np.allclose(D1_chara[idx], D2_chara[ith]) for ith, idx in enumerate(iso_mapping_Q)]):
            print('Find eqv', iso_mapping_Q, D1_chara, D2_chara) if test else None
            return True
    return False

    # method 2: check eqv rep using orthogonal theorem for each irrep
   #for ith, iso_mapping_Q in enumerate(sg_iso_Hinv['iso_mapping_Q']):
   #    for D2_mats in _permute_irrep(D2): 
   #        D1_mats = [D1.matrices[val] for val in iso_mapping_Q2PG.values()]
   #        D2_mats = [D2_mats[val] for val in iso_mapping_Q2PG.values()]
   #        if all([norm(D1_mats[idx] - D2_mats[ith]) < tol for ith, idx in enumerate(iso_mapping_Q)]):
   #            return True
   #    else:
   #        # if D1, D2 have different rep mat, use orthogonal theorem to check if they are equivalent
   #        D2_mats = [D2.matrices[val] for val in iso_mapping_Q2PG.values()]
   #        D1_chara = [D1.characters[val] for val in iso_mapping_Q2PG.values()]
   #        D1_chara_mapped = [D1_chara[idx] for idx in iso_mapping_Q]
   #        num_ir_D1 = sum([s == '_' for s in D1.label]) + 1 # 0 _: 3D, 1 _: 1+2D, 2 _: 1+1+1D 
   #        num_ir_D2 = sum([s == '_' for s in D2.label]) + 1
   #        assert  num_ir_D1 == num_ir_D2, (num_ir_D1, num_ir_D2)
   #        if check_rep_eqv_using_ortho_thm(D1_chara_mapped, D2_mats, num_ir_D2):
   #            assert all([np.allclose(c1, np.trace(m)) for c1, m in zip(D1_chara_mapped, D2_mats)]), ('Wrong!', D1_chara_mapped, D2_mats)
   #            # if D1, D2 are equivalent, return True
   #            print('Congratulation! Find equivalent 3D rep:', D1.label, D2.label)
   #            return True
   #return False

    # method 3: check eqv rep by explicitly computing the similarity transformation (problematic)
#   for ith, iso_mapping_Q in enumerate(sg_iso_Hinv['iso_mapping_Q']):
#       for D2_mats in _permute_irrep(D2): 
#           D1_mats = [D1.matrices[val] for val in iso_mapping_Q2PG.values()]
#           D2_mats = [D2_mats[val] for val in iso_mapping_Q2PG.values()]
#           if all([norm(D1_mats[idx] - D2_mats[ith]) < tol for ith, idx in enumerate(iso_mapping_Q)]):
#               return True
#           else:
#               # if D1, D2 have different rep mat, but same characters, check if there exists similarity transform 
#               D1_chara = [D1.characters[val] for val in iso_mapping_Q2PG.values()]
#               D2_chara = [D2.characters[val] for val in iso_mapping_Q2PG.values()]                   
#               if all([np.allclose(D1_chara[idx], D2_chara[ith]) for ith, idx in enumerate(iso_mapping_Q)]):
#                   D1_mats_mapped = [D1_mats[idx] for idx in iso_mapping_Q]
#                   block_dim, nir_list = None, None
#                   if sum([s == '_' for s in D1.label]) == 2:
#                       block_dim, nir_list = [1, 1, 1], [1, 1, 1]
#                   elif sum([s == '_' for s in D1.label]) == 1:
#                       block_dim, nir_list = [1, 2], [1, 1]
#                   try:
#                       C = rep_similarity_transformation(D1_mats_mapped, D2_mats, block_dim=block_dim, nir_list=nir_list)
#                       print('Conjugration! Find similarity transformation C for 3D rep:', D1.label, D2.label)
#                       return True
#                   except AssertionError as e:
#                       print('Warning: Fail to find C!, AssertionError=', e)
#   return False



def check_SSG_multitable(mtb_quot, spin_rot, iso_mapping_Q2PG):
    # check SSG satisfies group mutliplication table
    spin_op_sorted = [spin_rot[val] for val in iso_mapping_Q2PG.values()]
    mtb_spin = compute_mtb(spin_op_sorted)
    assert np.allclose(mtb_spin, mtb_quot), (mtb_spin, mtb_quot)


def find_quotgroup_uneqv_3Drep_bak(gid, quot_group_data, threeD_rep_list, iso_mapping_Q2PG, test=False, tol=1e-6):
    '''
    For a given quot group, find unequivalent 3D real reps. 
    
    Definition of equivalent 3D real reps under a given quot group: 
    two 3D real reps D1, D2 of quot group Q are equivalent if there exists an SG isomorphism W, 
    s.t. W @ (Q|D1) @ W^-1 = (Q|D2), 
    and W @ H @ W^-1 = H,  W @ T_H @ W^-1
    
    Method:
    1. first enumerate SG isomorphisms {W} that is compatible with H and T_H 
    2. find uneqv 3D reps
    '''
    # load sg data
    gid_data = SG_data[gid - 1]
    G_rotC, G_tauC = quot_group_data.G_rotC, quot_group_data.G_tauC
    H_rotC, H_tauC = quot_group_data.H_rotC, quot_group_data.H_tauC
    nop_G, nop_H = len(G_rotC), len(H_rotC)
   #sorted_G_rot, sorted_G_tau, sort_order = sort_rot(G_rotC, G_tauC)
    G_prim_basis = identify_SG_lattice(gid)[1] # each col is a prim basis vector
    T_H = quot_group_data.H_latt_PinP
    T_H_PinC = quot_group_data.H_latt_PinC
    
    H_rotP = [round_mat(inv(T_H_PinC) @ R @ T_H_PinC) for R in H_rotC]
    H_tauP = [inv(T_H_PinC) @ t for t in H_tauC]
    sorted_H_rotP, sorted_H_tauP, sort_order = sort_rot(H_rotP, H_tauP)

    # step 1: find sg isomorphism W compatible with inv subsg H and supercell T_H
    # => W @ H = H @ W, and T_H^-1 @ W @ T_H is an int matrix
    sg_iso_dict = find_sg_iso_transform(gid, test=0)
    sg_iso_Hinv = {'iso_rot': [], 'iso_mapping_G': [], 'iso_mapping_Q': []}
    for ith, iso_R in enumerate(sg_iso_dict['iso_rot']):
        iso_R_Hprim = round_mat(inv(T_H) @ iso_R @ T_H)  # iso_R_Hprim in H prim cell
        iso_R_Gconv = round_mat(G_prim_basis @ iso_R @ inv(G_prim_basis)) 
        if vec_is_int(iso_R_Hprim):  # iso_R in G prim cell
            # check if iso_R is an isomorphism of H, using H and G ops together:
            iso_mapping_G = sg_iso_dict['iso_mapping'][ith] # given by sorted G ops
            H_rot_idx = quot_group_data.H_op_idx
            H_dict = {'rot_index': H_rot_idx, 'tau_prim': [inv(T_H_PinC) @ t for t in H_tauC]}

            is_iso, iso_tau_Gprim = check_eqv_subsg(H_dict, H_dict, gid_data, T_H, 
                               sg_iso_trans={'iso_rot': [iso_R], 'iso_mapping': [iso_mapping_G]})
            if is_iso:
                print('find iso:', iso_R, iso_mapping_G) if test else None
                iso_tau_Gconv = G_prim_basis @ iso_tau_Gprim
                iso_mapping_Q = find_quot_mapping(quot_group_data, iso_R_Gconv, iso_tau_Gconv)
                sg_iso_Hinv['iso_rot'].append(iso_R)
                sg_iso_Hinv['iso_mapping_G'].append(iso_mapping_G)
                sg_iso_Hinv['iso_mapping_Q'].append(iso_mapping_Q) # given by quot group full op list

    # step 2: find uneqv 3D reps 
    # rep matrices need to align with quot_group full op list, using iso_mapping_Q2PG
    print('\n3D rep list:', [D.label for D in threeD_rep_list]) if test else None
    uneqv_rep_data = []
    for ith, rep in enumerate(threeD_rep_list):
        if (len(uneqv_rep_data) == 0 or
                all([not check_eqv_3Drep(rep, D, sg_iso_Hinv, iso_mapping_Q2PG, test=test) for D in uneqv_rep_data])):
            print('Find uneqv rep', rep.label) if test else None
            uneqv_rep_data.append(rep)
            # check multi table of spacial part and spin part, FIXME: skip for nonCryPG, as no rep matrices
           #check_SSG_multitable(quot_group_data.multi_table, rep.matrices, iso_mapping_Q2PG)

    return uneqv_rep_data 
    
    
def find_quotgroup_uneqv_3Drep(gid, quot_group_data, threeD_rep_list, iso_mapping_Q2PG, test=False, tol=1e-6):
    '''
    For a given quot group, find unequivalent 3D real reps. 
    
    Definition of equivalent 3D real reps under a given quot group: 
    two 3D real reps D1, D2 of quot group Q are equivalent if there exists an SG isomorphism W, 
    s.t. W @ (Q|D1) @ W^-1 = (Q|D2), 
    and W @ H @ W^-1 = H,  W @ T_H @ W^-1
    
    Method:
    1. First enumerate SG isomorphisms {W} that is compatible with H and T_H (iso_tau is solved using H)
    2. For two rep D1 and D2 that are isomorphic to the same PG, find mapping_Q of Q induced from D1 and D2
    2. check if there exist W s.t. W induces the same mapping_Q
    '''
    # load sg data
    gid_data = SG_data[gid - 1]
    G_rotC, G_tauC = quot_group_data.G_rotC, quot_group_data.G_tauC
    H_rotC, H_tauC = quot_group_data.H_rotC, quot_group_data.H_tauC
    nop_G, nop_H = len(G_rotC), len(H_rotC)
   #sorted_G_rot, sorted_G_tau, sort_order = sort_rot(G_rotC, G_tauC)
    G_prim_basis = identify_SG_lattice(gid)[1] # each col is a prim basis vector
    T_H = quot_group_data.H_latt_PinP
    T_H_PinC = quot_group_data.H_latt_PinC
    
    H_rotP = [round_mat(inv(T_H_PinC) @ R @ T_H_PinC) for R in H_rotC]
    H_tauP = [inv(T_H_PinC) @ t for t in H_tauC]
    sorted_H_rotP, sorted_H_tauP, sort_order = sort_rot(H_rotP, H_tauP)
    
    t0 = time()
    t_find_iso_tau = 0
    t_find_mapping = 0
    
    # step 1: find sg isomorphism W compatible with inv subsg H and supercell T_H
    # => W @ H = H @ W, and T_H^-1 @ W @ T_H is an int matrix
    sg_iso_dict = find_sg_iso_transform(gid, test=0)
    sg_iso_Hinv = {'iso_rot': [], 'iso_mapping_G': [], 'iso_mapping_Q': []}
    for ith, iso_R in enumerate(sg_iso_dict['iso_rot']):
        iso_R_Hprim = round_mat(inv(T_H) @ iso_R @ T_H)  # iso_R_Hprim in H prim cell
        iso_R_Gconv = round_mat(G_prim_basis @ iso_R @ inv(G_prim_basis)) 
        if vec_is_int(iso_R_Hprim):  # iso_R in G prim cell
            # check if iso_R is an isomorphism of H, using H and G ops together:
            iso_mapping_G = sg_iso_dict['iso_mapping'][ith] # given by sorted G ops
            H_rot_idx = quot_group_data.H_op_idx
            H_dict = {'rot_index': H_rot_idx, 'tau_prim': [inv(T_H_PinC) @ t for t in H_tauC]}

            tt0 = time()
            is_iso, iso_tau_Gprim_list = check_eqv_subsg(H_dict, H_dict, gid_data, T_H, 
                               sg_iso_trans={'iso_rot': [iso_R], 'iso_mapping': [iso_mapping_G]}, test=0)
            t_find_iso_tau += time() - tt0
            
            tt0 = time()
            if is_iso:
                print('find iso:', iso_R, iso_mapping_G) if test else None
                for iso_tau_Gprim in iso_tau_Gprim_list:
                    iso_tau_Gconv = G_prim_basis @ iso_tau_Gprim
                    iso_mapping_Q = find_quot_mapping(quot_group_data, iso_R_Gconv, iso_tau_Gconv)
                    if all([not np.allclose(iso_mapping_Q, tmp_map) for tmp_map in sg_iso_Hinv['iso_mapping_Q']]):
                        sg_iso_Hinv['iso_rot'].append(iso_R)
                        sg_iso_Hinv['iso_mapping_G'].append(iso_mapping_G)
                        sg_iso_Hinv['iso_mapping_Q'].append(iso_mapping_Q) # given by quot group full op list
            t_find_mapping += time() - tt0

    t1 = time()
    print('\n\n******* num sg_iso_rot:', len(sg_iso_dict['iso_rot']))
    print('******* num iso_mapping_Q:', len(sg_iso_Hinv['iso_mapping_Q']))
    print('******* time iso:%10.2f'%(t1 - t0))
    print('******* time find iso_tau:%10.2f'%(t_find_iso_tau))
    print('******* time find iso_mapping:%10.2f'%(t_find_mapping))

    t0 = time()
    # step 2: find uneqv 3D reps 
    # rep matrices need to align with quot_group full op list, using iso_mapping_Q2PG
    print('\n3D rep list:', [D.label for D in threeD_rep_list]) if test else None
    uneqv_rep_data = []
    for ith, rep in enumerate(threeD_rep_list):
        if (len(uneqv_rep_data) == 0 or
                all([not check_eqv_3Drep(rep, D, sg_iso_Hinv, iso_mapping_Q2PG, test=test) for D in uneqv_rep_data])):
            print('Find uneqv rep', rep.label) if test else None
            uneqv_rep_data.append(rep)
    threeD_rep_list = uneqv_rep_data

    print('******* time find uneqv 3D rep:%10.2f\n\n'%(time() - t0))



#   # sort 3D reps into groups that are isomorphic to the same pg
#   iso_pg_list = set([D.eqv_pg_label for D in threeD_rep_list])
#   rep_groups = {}
#   for pg_label in iso_pg_list:
#       rep_groups[pg_label] = [D for D in threeD_rep_list if D.eqv_pg_label == pg_label]
#   print('\n3D rep list:', rep_groups) if test else None

#   # step 2: for 3D reps isomorphic to the same pg, find mapping_Q induced from them
#   uneqv_rep_data = []
#   for pg_label, rep_list in rep_groups.items():
#       uneqv_rep_data_pg = [rep_list[0]]
#       if len(rep_list) >= 2:
#           for D1 in rep_list[1:]:
#               find_eqv_rep = False
#               for D2 in uneqv_rep_data_pg:
#                   # first find mapping_Q, for some eqv_pg, mapping_Q is not unique
#                   print('checking eqv rep for ', D1.label, D2.label)#if test else None
#                   mapping_Q_list = find_mapping_Q_for_two_rep(D1, D2, quot_group_data, iso_mapping_Q2PG)
#                   # then check if there exist sg isomorphism W={A|t} which induces mapping_Q, i.e., solve t
#                   if check_mapping_is_isomorphism(mapping_Q_list, sg_iso_Hinv, quot_group_data):
#                       # if W exists, then D1 and D2 are equivalent
#                       find_eqv_rep = True

#               # if no equivalent rep is find, then add D1 to uneqv data of pg
#               if not find_eqv_rep: 
#                   uneqv_rep_data_pg.append(D1)
#       
#       uneqv_rep_data.extend(uneqv_rep_data_pg)

    return uneqv_rep_data 

    

def find_mapping_Q_for_two_rep(D1, D2, quot_group_data, iso_mapping_Q2PG):
    # find mapping_Q given by D1 and D2, which are two 3D real reps isomorphic to the same PG
    def check_rot_list_eqv(rot_list1, rot_list2):
        eqv = True
        mapping_D1D2 = [] 
        for i1, R1 in enumerate(rot_list1): 
            idx_list = [i2 for i2, R2 in enumerate(rot_list2) if np.allclose(R1, R2)]
            if len(idx_list) != 1:
                eqv = False
            else:
                idx = idx_list[0]
                mapping_D1D2.append(idx)
        return eqv, mapping_D1D2
    assert D1.eqv_pg_label == D2.eqv_pg_label and len(D1.matrices_real) == len(D2.matrices_real)

    # first find mapping_D1D2, by comparing rep matrices directly, as D1, D2 have the same main axis along z
    # but for D.eqv_pg_label of some PGs, mapping is not unique, need to consider all possible mapping_Q
    # Ex: mmm, Cn (change Cn and Cn^-1), Cnh, Sn, etc.

   # get all possible permutations of irreps in D2. Use D2.matrices which can be permuted directly, but not D2.matrices_real (can not permute directly)
    D2_mat_permuted_list = [D2.matrices]
    if len(D2.label.split('_')) == 3: # 1+1+1D
        order_list = [[0,2,1], [1,0,2], [1,2,0], [2,0,1], [2,1,0]]
        for order in order_list:
            D2_mat_permuted_list.append([np.diag((Dg.diagonal()[order[0]], Dg.diagonal()[order[1]], Dg.diagonal()[order[2]])) for Dg in D2.matrices])
        
    elif len(D2.label.split('_')) == 2: # 1+1+1D
        D2_mat_permuted_list.append([mat_direct_sum(np.array(Dg[2,2]).reshape((1,1)), Dg[0:2, 0:2]) for Dg in D2.matrices])

    mapping_D1D2_list = []
    for ith, D2mat in enumerate(D2_mat_permuted_list):
        eqv, mapping = check_rot_list_eqv(D1.matrices, D2mat)
        if eqv:
            if all([not np.allclose(m1, mapping) for m1 in mapping_D1D2_list]):
                mapping_D1D2_list.append(mapping)
   #assert len(mapping_D1D2_list) >= 1, (mapping_D1D2_list)
     
    # transform mapping_D1D2 to mapping_Q, using iso_mapping_Q2PG
    mapping_Q_list = [] 
    for mapping_D1D2 in mapping_D1D2_list:
        mapping_Q = {}
        for iQ1, iPG in iso_mapping_Q2PG.items():
            iPG2 = mapping_D1D2[iPG]
            iQ2 = [i2 for i2, j2 in iso_mapping_Q2PG.items() if j2 == iPG2][0]
            mapping_Q[iQ1] = iQ2
        mapping_Q_list.append(mapping_Q)

    return mapping_Q_list


def check_mapping_is_isomorphism(mapping_Q_list, sg_iso_Hinv, quot_group_data, tol=1e-6, test=False):
    # for input mapping_Q, check if it can be given by any isomorphism W={A|t} in sg_iso_Hinv
    
    G_rotC_list, G_tauC_list, sort_order = sort_rot(quot_group_data.G_rotC, quot_group_data.G_tauC)
    G_prim_basis = identify_SG_lattice(quot_group_data.G_gid)[1] # each col is a prim basis vector
    G_rotP_list = [inv(G_prim_basis) @ R @ G_prim_basis for R in G_rotC_list]
    G_tauP_list = [inv(G_prim_basis) @ t for t in G_tauC_list]
    nop_G = len(G_rotC_list)

    Q_rotC, Q_tauC = quot_group_data.rotC_fulllist, quot_group_data.tauC_fulllist
    nop_Q = len(Q_rotC)
    S = quot_group_data.H_latt_PinP # supercell
    Q_rotP = [round_mat(inv(G_prim_basis @ S) @ R @ (G_prim_basis @ S)) for R in Q_rotC] # in Q prim latt, use supercelll S, inv(S) @ R @ S, from G prim to Q prim
    Q_tauP = [inv(S) @ t for t in Q_tauC]

    for mapping_Q in mapping_Q_list:
        Q_rotC_mapped = [Q_rotC[ith] for ith in mapping_Q.values()]
        Q_tauC_mapped = [Q_tauC[ith] for ith in mapping_Q.values()]
        Q_rotP_mapped = [Q_rotP[ith] for ith in mapping_Q.values()]
        Q_tauP_mapped = [Q_tauP[ith] for ith in mapping_Q.values()]
        assert len(mapping_Q) == len(Q_rotC)
        
        for iso_R_Gprim, W_mapping_G, W_mapping_Q in zip(sg_iso_Hinv['iso_rot'], sg_iso_Hinv['iso_mapping_G'], sg_iso_Hinv['iso_mapping_Q']):
            # if W_mapping_Q is the same as mapping_Q, then D1, D2 are equivalent
            if all([v1 == v2 for v1, v2 in zip(mapping_Q.values(), W_mapping_Q)]):
                return True
            else:
                # First check if mapping_Q of rotation parts can be given by iso_R
                iso_R_Gconv = round_mat(G_prim_basis @ iso_R_Gprim @ inv(G_prim_basis)) 
                iso_R_Qprim = round_mat(inv(S) @ iso_R_Gprim @ S)
                Q_rotC_transformed = [round_mat(iso_R_Gconv @ R @ inv(iso_R_Gconv)) for R in Q_rotC]
                if all([np.allclose(R1, R2) for R1, R2 in zip(Q_rotC_transformed, Q_rotC_mapped)]):
                    # Second, solve iso_tau, using W G W^-1 = G, W Q W^-1 = Q_mapped
                    
                    # transform W G = G W and W Q = Q_mapped W into M @ t = w % [1]
                    # ==> (R2 - 1) S t = iso_R @ t1 - t2, for op in G, written in G prim latt, t in Q prim latt
                    #     (R2 - 1) t = S^-1 @ iso_R @ S @ t1 - t2, for op in Q, Q_mapped,
                    # written in Q prim latt, where S is supercell, S^-1 @ iso_R @ S is iso_R in Q prim latt  
                    # solution t is in Q prim latt
                    M1 = np.zeros((3 * nop_G, 3))
                    w1 = np.zeros(3 * nop_G)
                    for iop, R1, t1, iop2 in zip(range(nop_G), G_rotP_list, G_tauP_list, W_mapping_G):
                        R2 = G_rotP_list[iop2]
                        t2 = G_tauP_list[iop2]
                        M1[iop * 3: (iop + 1) * 3, :] = (R2 - np.eye(3)) @ S
                        w1[iop * 3: (iop + 1) * 3] = iso_R_Gprim @ t1 - t2

                    M2 = np.zeros((3 * nop_Q, 3))
                    w2 = np.zeros(3 * nop_Q)
                    for iop, R1, t1, R2, t2 in zip(range(nop_Q), Q_rotP, Q_tauP, Q_rotP_mapped, Q_tauP_mapped):
                        M2[iop * 3: (iop + 1) * 3, :] = (R2 - np.eye(3))
                        w2[iop * 3: (iop + 1) * 3] = iso_R_Qprim @ t1 - t2 

                    M = np.vstack((M1, M2))
                    w = np.hstack((w1, w2))
                    print('iso_R', iso_R_Gprim, iso_R_Qprim) if test else None
                    print('M1, w1', M1, w1) if test else None
                    print('M2, w2', M2, w2) if test else None

                    # solve M @ t = w % [1]
                    trial_tlist = solve_linear_inhomogeneous(M, w, mod_num=[1,1,1], allow_infinite_sol=True)
                    if len(trial_tlist) > 0:
                        iso_tau_Qprim = trial_tlist[0]
                        iso_tau_Gprim = S @ iso_tau_Qprim
                        return True#, iso_tau_Gprim
    return False 




if __name__ == '__main__':

    for gid in [3]:
        print('\n\n\ngid:', gid)
        iso_trans = find_sg_iso_transform(gid, test=True)
       #print(iso_trans)
