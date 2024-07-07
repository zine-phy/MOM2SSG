import numpy as np
from sympy import *
from mapping_data1 import mapping_data1
from itertools import product
 
def ppr(matr, left):
    pprint([left, matr])
    print("-----------------------------------------------")
 
def mat_is_diagonal(mat):
    mat = np.array(mat).astype(int)
    for s in range(min(np.shape(mat))):
        mat[s,s] = 0
    return True if np.max(np.abs(mat)) == 0 else False


def num_rank(num, Z):
    # return the rank of num in Zn group
    assert num % 1 == 0 and 0 <= num < Z, (num, Z)
    for time in range(1,Z+1):
        if (num * time) % Z == 0:
            return time

def find_least_num_of_same_rank(num, Z):
    # find the least num that has the same rank with num in Zn group.
    if num in [0, 1]:
        return num, 1
   
    rank0 = num_rank(num, Z)
    least_num, time = num, 1
    for tmp_time in range(1,Z):
        tmp_num = (tmp_time * num) % Z
        tmp_rank = num_rank(tmp_num, Z)
        if tmp_rank == rank0 and tmp_num < least_num:
            least_num, time = tmp_num, tmp_time
    return least_num, time
            
def find_multiple(n1, n2, Z):
    # for two number n1, n2, find the multiple m s.t n1 - m * n2 % Z is smallest

    least_num, time = n1, 1
    for tmp_time in range(1,Z):
        tmp_num = (n1 - tmp_time * n2) % Z
        tmp_rank = num_rank(tmp_num, Z)
        if tmp_num < least_num:
            least_num, time = tmp_num, tmp_time
    return time

            
# Moves least element in south-western block, that starts at position [s, s] into start position
def move_least_to_start(matr: Matrix, left: Matrix, s: int, col_shift: Matrix, mod_num: Matrix):
    rows, cols = len(matr.col(0)), len(matr.row(0))
    
    while all([ matr[i, s + col_shift[0]] == 0 for i in range(s, rows) ]):
        col_shift[0] += 1

    s1 = s + col_shift[0]
    pos = [s, s1]
    rank = num_rank(matr[s, s1], mod_num[s1])
    for i in range(s, rows):
        if matr[i, s1] != 0 and (num_rank(matr[i, s1], mod_num[s1]) > rank or (num_rank(matr[i, s1], mod_num[s1]) == rank and matr[i, s1] < matr[s, s1])):
            pos = [i, s1]
            rank = num_rank(matr[i, s1], mod_num[s1])
 
    if pos[0] > s:
        matr.row_swap(s, pos[0])
        left.row_swap(s, pos[0])
 
        
 
# Modifies edging by standard operations in order to make it zero, should be followed by null_edging()
def modify_edging(matr: Matrix, left: Matrix, s: int, col_shift: Matrix, mod_num: Matrix):
    rows, cols = len(matr.col(0)), len(matr.row(0))
    s1 = s + col_shift[0]
 
   #for i in range(s + 1, rows):
   #    if matr[i, s] != 0:
    for i in range(rows):
        if i != s and matr[i, s1] != 0:
            assert num_rank(matr[i,s1], mod_num[s1]) <= num_rank(matr[s, s1], mod_num[s1]),(matr[i,s1], matr[s,s1])
            multiple = find_multiple(matr[i, s1], matr[s, s1], mod_num[s1])
            matr.row_op(i, lambda val, col: (val - multiple * matr[s, col]) % mod_num[col])
            left.row_op(i, lambda val, col: val - multiple * left[s, col])
            

 
 
# Moves least element in edging, that starts at position [s, s] into start position
def move_le_to_start(matr: Matrix, left: Matrix, s: int, col_shift: Matrix, mod_num: Matrix):
    rows, cols = len(matr.col(0)), len(matr.row(0))
    s1 = s + col_shift[0]
    pos = [s, s1]
    num, rank = matr[s, s1], num_rank(matr[s, s1], mod_num[s1])
 
    for i in range(s + 1, rows):
        if matr[i, s1] != 0 and (num_rank(matr[i, s1], mod_num[s1]) > rank or (num_rank(matr[i, s1], mod_num[s1]) == rank and matr[i, s1] < matr[s, s1])):
            pos = [i, s1]
            rank = num_rank(matr[i, s1], mod_num[s1]) 
 
    if pos[0] > s:
        matr.row_swap(s, pos[0])
        left.row_swap(s, pos[0])
 

# Makes edging, that starts at pos [s, s] zero, ensures south-western block does not divide element at pos [s, s]
def null_edging(matr: Matrix, left: Matrix, s: int, col_shift: Matrix, mod_num: Matrix):
    rows, cols = len(matr.col(0)), len(matr.row(0))
 
    while not matr[s + 1: rows, s + col_shift[0]].is_zero_matrix:
        move_le_to_start(matr, left, s, col_shift, mod_num)
        modify_edging(matr, left, s, col_shift, mod_num)
        
        ppr(matr, left)
 
 
 
# Iteration of transformation into Smith normal form, modifies edging starting at position [s, s]
def transform_smith(matr: Matrix, left: Matrix, s: int, col_shift: Matrix, mod_num: Matrix, test = False):
    move_least_to_start(matr, left, s, col_shift, mod_num)
    print('first:') if test else None
    ppr(matr, left) if test else None
 
    modify_edging(matr, left, s, col_shift, mod_num)
    print('second:') if test else None
    ppr(matr, left) if test else None
 
    null_edging(matr, left, s, col_shift, mod_num)
    print('third:') if test else None
    ppr(matr, left) if test else None
 
 
# Checks whether south-western block of matrix, starting at pos [s, s], is zero
def next_block_empty_or_null(matr: Matrix, s: int, col_shift: Matrix, mod_num: Matrix):
    rows, cols = len(matr.col(0)), len(matr.row(0))
    return matr[s + 1: rows, s + 1 + col_shift[0]: cols].is_zero_matrix
 
 
def smith_form(matr: Matrix, mod_num: Matrix, test = False):
    # Computes the modified Smith's form B of matrix A, such that L * A = B
    # returns: - matrix in Smith form (B)  - left square matrix (L)
    orig_mat = matr.copy()
    matr = matr.copy()
    rows, cols = len(matr.col(0)), len(matr.row(0))
    left = Matrix.eye(rows)
    ppr(matr, left)
 
    if matr.is_zero_matrix:
        return matr, left
    
    col_shift = Matrix([0])
    for s in range(0, min(rows, cols)):
        transform_smith(matr, left, s, col_shift, mod_num, test)
 
        if next_block_empty_or_null(matr, s, col_shift, mod_num):
            break

    ppr(matr, left)
   #assert mat_is_diagonal(matr), matr
    assert all([[ num % mod == 0 for num, mod in zip(tmp_row, mod_num) ] for tmp_row in np.array(left * orig_mat - matr).astype('int') ]), left * orig_mat
    return matr, left


def smith_form_np(mat, mod_num):
    SM, L = smith_form(Matrix([ [int(a) for a in line] for line in mat ]), Matrix(mod_num), test = 0)
    return np.array(SM).astype(int), np.array(L).astype(int)

 