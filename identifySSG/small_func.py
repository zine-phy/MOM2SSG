import numpy as np
from numpy.linalg import norm, inv, eigh, det
from itertools import product
# from smith_form.gauss_elim import gauss_elim_np
# from find_minimal_latt import standardize_prim_basis
from .smith_form.smith_form_C import smith_form
import sympy as sp


def latt_home(vec,tol=1e-6):
    vec = np.array([ np.round(v) if norm(v - np.round(v))<tol else v for v in vec]) # first take round, in case of number like 0.9999
    return vec - np.floor(vec)

def find_lcm(x, y):
    greater = x if x > y else y  
    while True:
        if greater % x == 0 and greater % y == 0:
            lcm = greater
            break
        greater += 1
    return lcm 

def round_vec(vec, tol=1e-6):
    return np.array([np.round(v) if abs(v - np.round(v)) < tol else v for v in vec]) 

def vec_is_int(vec, tol=1e-6):
    return True if norm(vec - np.round(vec)) < tol else False

def round_num(num, tol=1e-6):
    return int(round(num)) if abs(num - round(num)) < tol else num
    
def round_mat(m, tol=1e-6):
    return np.array(np.round(m), dtype=int) if norm(m - np.round(m)) < tol else m

def find_mat(m, m_list, tol=1e-6):
    tmp = [norm(m - R) < tol for R in m_list]
    assert sum(tmp) == 1, (m, m_list)
    return tmp.index(1)

def mat_in_list_int(M, _list):
    # tell if int mat M is in an int mat list
    M = round_mat(M)
    return True if any([np.allclose(M, N) for N in _list]) else False

def mat_in_list(M, _list):
    # tell if  mat M is in mat list
    return True if any([norm(M - N) < 1e-6 for N in _list]) else False

def trans_in_group(trans, latt):
    # check if trans is an integer combination of latt, each col of latt is a latt vec
    sol = np.linalg.solve(latt, trans)
   #print('sol', sol, trans)
    return True if vec_is_int(sol) else False

def op_prod(R1, t1, R2, t2):
    # return (R1|t1)(R2|t2) = (R1 @ R2, R1 @ t2 + t1)
    return R1 @ R2, R1 @ t2 + t1

def op_inv(R, t):
    return inv(R), -inv(R) @ t

def mat_direct_sum(M1, M2):
    r1, c1 = np.shape(M1)
    r2, c2 = np.shape(M2)
    M = np.zeros((r1 + r2, c1 + c2), dtype=complex)
    M[:r1, :c1] = M1
    M[r1: r1 + r2, c1: c1 + c2] = M2
    return M

def compute_mtb(ops, tol=1e-3):
    # get multiplication table
    nop = len(ops)
    mtb = np.zeros((nop, nop), dtype=int)
    for i1, R1 in enumerate(ops):
        for i2, R2 in enumerate(ops):
            mtb[i1, i2] = [i3 for i3, R3 in enumerate(ops) if norm(R3 - R1 @ R2) < tol][0]
    return mtb


def print_mat(mat, wfile=None, print1=True):
    dim = np.shape(mat)[0]
    strg = ''
    for row in range(dim):
        strg += '[ '
        for col in range(dim):
            num = mat[row, col]
            if abs(num) < 1e-8:
                strg += '     0      '
            elif abs(num.real) < 1e-8:
                strg += ' %4.3f *I ' % (num.imag)
            elif abs(num.imag) < 1e-8:
                strg += ' %4.3f ' % (num.real)
            else:
                strg += ' (%4.3f + %4.3f *I) ' % (num.real, num.imag)
        strg += ' ]\n'
    if print1:
        print(strg, file=wfile)
    else:
        return strg



def find_Bravais_prim_vec(latt):
    prim_latt = {'P' : np.eye(3),
                 'B' : np.array([[1/2, -1/2, 0], [1/2, 1/2, 0], [0, 0, 1]]),
                 'B2': np.array([[1, 0, 0], [0, 1/2, 1/2], [0, -1/2, 1/2]]),
                 'B3': np.array([[1/2, 0, -1/2], [0, 1, 0], [1/2, 0, 1/2]]),
                 'F' : np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2,
                 'I' : np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 2,
                 'R' : np.array([[2, 1, 1], [-1, 1, 1], [-1, -2, 1]]) / 3
                }
    assert latt in prim_latt.keys(), latt
    return prim_latt[latt]

def find_prim_latt_trans_spglib(latt):
    if latt == 'P':
        return []
    elif latt == 'B':
        return [np.array([0.5, 0.5, 0])]
    elif latt == 'B2':
        return [np.array([0, 0.5, 0.5])]
    elif latt == 'B3':
        return [np.array([0.5, 0, 0.5])]
    elif latt == 'I':
        return [np.array([0.5, 0.5, 0.5])]
    elif latt == 'F':
        return [np.array([0, 0.5, 0.5]), np.array([0.5, 0, 0.5]), np.array([0.5, 0.5, 0])]
    elif latt == 'R':
        return [np.array([2/3, 1/3, 1/3]), np.array([1/3, 2/3, 2/3])]
    else:
        raise ValueError('Wrong latt!', latt)

def find_Bravais_cellmultiple(latt):
    cell_multiple = {'P':1, 'B':2, 'B2':2, 'B3':2, 'F':4, 'I':2, 'R':3}
    assert latt in cell_multiple.keys(), latt
    return cell_multiple[latt]


def write_subsg_index(supsg, subsg, t_index, k_index, subsg_dict):
    if len(subsg_dict['subsg']) > 0 and any([subsg == g and t_index == t and k_index == k 
            for g, t, k in zip(subsg_dict['subsg'], subsg_dict['t_index'], subsg_dict['k_index'])]):
        return subsg_dict
    elif subsg == supsg and t_index == 1 and k_index == 1:
        # same as supsg, not a subgroup
        return subsg_dict
    else:
        subsg_dict['subsg'].append(subsg)
        subsg_dict['t_index'].append(t_index)
        subsg_dict['k_index'].append(k_index)
        return subsg_dict

def check_subsg_result(subsg_dict, path):
    # check if computed subsg_dict agree with bilbao subsg data
    # method: check if number of subsg is equal, and if all subsg in subsg_dict also in bilbao_subsg, this ensures two dicts are identical.
    gid = subsg_dict['gid']
    bilbao_subsg_data = np.load('%s/data/subsg_index_data.npy'%path, allow_pickle=True)
    bilbao_dict = bilbao_subsg_data[gid - 1]
    print('\n\nSG %d  result subsg dict:  num subsg: %d\n'%(gid, len(subsg_dict['subsg'])), subsg_dict)
    print('bilbao subsg dict:  num subsg: %d\n'%(len(bilbao_dict['subsg'])), bilbao_dict)
    for g, t, k in zip(subsg_dict['subsg'], subsg_dict['t_index'], subsg_dict['k_index']):
        if sum([g == bg and t == bt and k == bk for bg, bt, bk in zip(bilbao_dict['subsg'], bilbao_dict['t_index'], bilbao_dict['k_index'])]) == 1:
            continue
        else:
            print('Extra subsg!  subsg: %d, t-index: %d, k-index: %d'%(g, t, k))
    for g, t, k in zip(bilbao_dict['subsg'], bilbao_dict['t_index'], bilbao_dict['k_index']):
        if sum([g == bg and t == bt and k == bk for bg, bt, bk in zip(subsg_dict['subsg'], subsg_dict['t_index'], subsg_dict['k_index'])]) == 1:
            continue
        else:
            print('Omitted subsg!  subsg: %d, t-index: %d, k-index: %d'%(g, t, k))
   #assert bilbao_dict['gid'] == gid and len(bilbao_dict['subsg']) == len(subsg_dict['subsg']), (
   #        bilbao_dict, subsg_dict, len(bilbao_dict['subsg']), len(subsg_dict['subsg']))   
   #for g, t, k in zip(subsg_dict['subsg'], subsg_dict['t_index'], subsg_dict['k_index']):
   #    if sum([g == bg and t == bt and k == bk for bg, bt, bk in zip(bilbao_dict['subsg'], bilbao_dict['t_index'], bilbao_dict['k_index'])]) == 1:
   #        continue
   #    else:
   #        raise ValueError('Wrong subsg!', g, t, k, bilbao_dict)
   #return True


def find_monoclinic_trial_iso(Max=50):
    # for monoclinic SG 3-15, give possible isomorphisms
   #trial_iso = []    
   #for a11 in range(-Max, Max+1):
   #    for a13 in range(-Max, Max+1):
   #        for a31 in range(-Max, Max+1):
   #            for a33 in range(-Max, Max+1):
   #                S = np.array([[a11, 0, a13],[0, 1, 0],[a31, 0, a33]])
   #                if a11 * a33 - a13 * a31 == 1:
   #                    trial_iso.append(S)

    trial_iso = [np.array([[comb[0], 0, comb[1]],[0, 1, 0],[comb[2], 0, comb[3]]], dtype=int)
                      for comb in product(*np.tile(np.arange(-Max, Max + 1), 4).reshape((4, 2*Max+1)))
                      if comb[0] * comb[3] - comb[1] * comb[2] == 1]
    return trial_iso

def det3(M):
    # determinent of 3*3 matrix, return integer
    return (M[0,0]*M[1,1]*M[2,2] - M[0,0]*M[1,2]*M[2,1] + M[0,1]*M[1,2]*M[2,0] 
            - M[0,1]*M[1,0]*M[2,2] + M[0,2]*M[1,0]*M[2,1] - M[0,2]*M[1,1]*M[2,0])

def find_triclinic_trial_iso():
    # for triclinic SG 1-2, give possible isomorphisms. 
    # For SL(3, Z) matrices, Max=2 ==> 67704. Max=3 ==> N=640824
    Max = 2
   #trial_iso1 = [np.array(comb, dtype=int).reshape((3,3))
   #                  for comb in product(*np.tile(np.arange(0, Max + 1), 9).reshape((9, Max + 1)))
   #                  if det3(np.array(comb).reshape((3,3))) == 1]
    trial_iso1 = [np.array(comb, dtype=int).reshape((3,3))
                      for comb in product(*np.tile(np.arange(-Max, Max + 1), 9).reshape((9, 2*Max+1)))
                      if det3(np.array(comb).reshape((3,3))) == 1]

    # consider SL(2, Z) matrices along y,z axis, as supercells in triclinic systems are along y,z direction
    Max = 24 # 12 
    trial_iso2 = [np.array([[1, 0, 0], [0, comb[0], comb[1]], [0, comb[2], comb[3]]], dtype=int)
                      for comb in product(*np.tile(np.arange(-Max, Max + 1), 4).reshape((4, 2*Max+1)))
                      if comb[0] * comb[3] - comb[1] * comb[2] == 1]
    
    return trial_iso1 + trial_iso2

def float_to_sympy(a, tol=1e-8):
    # convert a real float number to sympy number
    sign = 1 if abs(a) == a else -1
    a = abs(a)
    if abs(a - np.round(a)) < tol: # integer
        return int(np.round(a)) * sign
    elif abs(a - 0.5) < tol:
        return sp.Rational(1/2) * sign

    elif abs(3 * a - np.round(3 * a)) < tol:
        return sp.Rational(np.round(3 * a) / 3).limit_denominator(100) * sign
    elif abs(4 * a - np.round(4 * a)) < tol:
        return sp.Rational(np.round(4 * a) / 4).limit_denominator(100) * sign
    elif abs(5 * a - np.round(5 * a)) < tol:
        return sp.Rational(np.round(5 * a) / 5).limit_denominator(100) * sign
    elif abs(6 * a - np.round(6 * a)) < tol:
        return sp.Rational(np.round(6 * a) / 6).limit_denominator(100) * sign
    elif abs(8 * a - np.round(8 * a)) < tol:
        return sp.Rational(np.round(8 * a) / 8).limit_denominator(100) * sign
    elif abs(10 * a - np.round(10 * a)) < tol:
        return sp.Rational(np.round(10 * a) / 10).limit_denominator(100) * sign
    elif abs(12 * a - np.round(12 * a)) < tol:
        return sp.Rational(np.round(12 * a) / 12).limit_denominator(100) * sign
        
    elif abs(a - 0.8660254038) < tol:
        return sp.sqrt(3)/2 * sign 
    elif abs(a - 0.7071067812) < tol:
        return sp.sqrt(2)/2 * sign 
    elif abs(a - 0.6830127019) < tol:
        return (sp.sqrt(3) + 1)/4 * sign
    elif abs(a - 0.1830127019) < tol:
        return (sp.sqrt(3) - 1)/4 * sign
    else:
        raise ValueError('Wrong float,a='+str(a))

def numpy_array_to_sympy(A):
    # convert a numpy array to sympy matrix
    shape = np.shape(A)
    if len(shape) == 1:
        B = sp.zeros(shape[0], 1)
        shape = [shape[0], 1]
        A = A.reshape(shape)
    else:
        B = sp.zeros(*shape)
        
    for i in range(shape[0]):
        for j in range(shape[1]):
            B[i,j] = float_to_sympy(np.real(A[i,j])) + sp.I * float_to_sympy(np.imag(A[i,j]))
    return B


def solve_linear_inhomogeneous(M, w, mod_num=[1,1,1], tol=1e-6, allow_infinite_sol=False, test=False):
    # find v s.t. M @ v = w mod n
    # 1. compute smith form, i.e., M = L^-1 @ SM @ R^-1
    # 2. compute all possible v, s.t. SM @ v1 = L @ w, and SM @ R^-1 @ v = L @ w (v=R@v1)

    assert np.allclose(M, np.array(M).astype(int)), M
    M = np.array(M).astype(int)

    rank, SM, L, LI, R, RI = smith_form(M)
    # rank, SM, L, LI, R, RI = smith_form_sage(M)

    diag = SM.diagonal()
    w_prime = L @ w
    nop = len(diag) // 3
    mod_num_all = mod_num * (np.shape(w)[0] // 3)  # mod_num for each vi
    print('M, w:\n', M, w, '\ndiag', SM.diagonal()) if test else None 
    print('w_prime', w_prime) if test else None
    print('R', R) if test else None
    assert np.allclose(M, LI @ SM @ RI), (M, LI, SM, RI)

    if not all([round_num(wi) % mod_i == 0 for irow, wi, mod_i in zip(range(len(w_prime)), w_prime, mod_num_all) 
                if irow >= len(diag) or diag[irow] == 0]):
        # no soluation for Mv=w
       #print('mod 1 of w_prime')
       #print([round_num(wi) % mod_i  for irow, wi, mod_i in zip(range(len(w_prime)), w_prime, mod_num_all) 
       #        if irow >= len(diag) or diag[irow] == 0])
        return []

    if allow_infinite_sol:
        # if allow infinite solutions, then the diagonal elements can be zero, 
        # and are replaced by 1 to obtain finite solutions.
        diag = [1 if d == 0 else d for d in diag]
    else:
        assert all([d > 0 for d in diag]), (diag, M, w)

    vplist = [] # SM @ vp = w_prime
    for combination in product(*[range(d) for d in diag]):
        v = np.array([float(wi) / di + mod_i * ci / di for ci, di, wi, mod_i in zip(combination, diag, w_prime, mod_num_all)])
        vplist.append(v)

    vlist = [round_vec(R @ v) for v in vplist]
    assert all([norm(np.mod(round_vec(M @ v - np.array(w).astype(float).reshape(np.shape(w)[0])), mod_num_all)) < tol for v in vlist]),\
                (vlist, [round_vec(M @ v - w) for v in vlist], mod_num,
                 [round_vec(SM @ vp - w_prime) for vp in vplist])
    return vlist


def show_linear_inhomogeneous_sp(M, w):
    rank, SM, L, LI, R, RI = smith_form_sage(M)
    diag = SM.diagonal()
    w_prime = L @ w
    print('M, w:\n', M, w, '\ndiag', SM.diagonal()) 
    print('w_prime', w_prime, len(diag))


def test():
    t = np.array([[0,1,0], [0,0,1], [0,1,1]])
    t_p = np.array([[0,1,0], [0,1,1], [0,0,1]])
    M = np.zeros((9,9), dtype=int)
    for ith in range(3):
        for jth in range(3):
            M[3 * ith + jth, 3 * jth: 3 * (jth + 1)] = t[ith]
    w = t_p.flatten() 
   
    vlist = solve_linear_inhomogeneous(M, w, mod_num=[1,2,2], allow_infinite_sol=True, test=1) 
    print('\nsol', vlist)


def test_sympy():
    x =sp.symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9', int=True)
    M = sp.Matrix([[x[0], 0, 0], [x[1] + 2, 1, -1], [x[2], 0, 1]])
   #M = sp.Matrix([[0, 0, 0], [x[1] + 2, 1, -1], [x[2], 0, 1]])
    sol = sp.solve(sp.det(M) - 1, x)
    print(sol)








if __name__ == '__main__':

   #test()
   #test_sympy()
    iso = find_triclinic_trial_iso(Max=36)
    print(len(iso))