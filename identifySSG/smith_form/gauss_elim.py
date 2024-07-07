import numpy as np
from sympy import *
 
def ppr(matr, left):
    pprint([left, matr])
    print("-----------------------------------------------")
 
def find_lcm(x, y):
    greater = x if x > y else y  
    while True:
        if greater % x == 0 and greater % y == 0:
            lcm = greater
            break
        greater += 1
    return lcm 
 
# Moves least element in south-western block, that starts at position [s, s] into start position
def move_least_to_start(matr: Matrix, left: Matrix, s: int):
    rows, cols = len(matr.col(0)), len(matr.row(0))
 
    pos = [s, s]
    num = abs(matr[s, s])
    for i in range(s, rows):
        if matr[i, s] != 0 and (num == 0 or abs(matr[i, s]) < num):
            pos = [i, s]
            num = abs(matr[i, s])
 
    if pos[0] > s:
        matr.row_swap(s, pos[0])
        left.row_swap(s, pos[0])
 
    if matr[s, s] < 0:
        matr.row_op(s, lambda val, col: -val)
        left.row_op(s, lambda val, col: -val)
 
 
# Modifies edging by standard operations in order to make it zero, should be followed by null_edging()
def modify_edging(matr: Matrix, left: Matrix, s: int):
    rows, cols = len(matr.col(0)), len(matr.row(0))
 
    for i in range(s + 1, rows):
        if matr[i, s] != 0:
            q = matr[i, s] // matr[s, s]
            matr.row_op(i, lambda val, col: val - q * matr[s, col])
            left.row_op(i, lambda val, col: val - q * left[s, col])
 
 
# Moves least element in edging, that starts at position [s, s] into start position
def move_le_to_start(matr: Matrix, left: Matrix, s: int):
    rows, cols = len(matr.col(0)), len(matr.row(0))
    pos = [s, s]
    num = abs(matr[s, s])
 
    for i in range(s + 1, rows):
        if matr[i, s] != 0 and abs(matr[i, s]) < num:
            pos = [i, s]
            num = abs(matr[i, s])
 
    if pos[1] == s and pos[0] > s:
        matr.row_swap(s, pos[0])
        left.row_swap(s, pos[0])
 
 
# Makes edging, that starts at pos [s, s] zero, ensures south-western block does not divide element at pos [s, s]
def null_edging(matr: Matrix, left: Matrix, s: int):
    rows, cols = len(matr.col(0)), len(matr.row(0))
 
    while not matr[s + 1: rows, s].is_zero_matrix:
        move_le_to_start(matr, left, s)
        modify_edging(matr, left, s)
 
    if matr[s, s] < 0:
        matr.row_op(s, lambda val, col: -val)
        left.row_op(s, lambda val, col: -val)
 
# Modifies upper edging by standard operations in order to make it zero. Upper rows is allowed to time num > 1
def modify_upper_edging(matr: Matrix, left: Matrix, s: int, out='prim'):
    rows, cols = len(matr.col(0)), len(matr.row(0))
 
    for i in range(s):
        if matr[i, s] != 0:
            #assert matr[i, s] % matr[s, s] == 0, (matr[i, s], matr[s, s], matr)
            # matr[i,s] may not be a multiple of matr[s,s]
            q = matr[i, s] // matr[s, s]  # if matr[i, s] < matr[s, s], then q=0
            matr.row_op(i, lambda val, col: val - q * matr[s, col])
            left.row_op(i, lambda val, col: val - q * left[s, col])
            
            if matr[i, s] != 0 and out == 'conv': # null upper rows to get conv basis, allowing to time num>1
                # after previous step, matr[i,s] < matr[s,s]. Use lcm to null matr[i, s]
                lcm = find_lcm(matr[s, s], matr[i, s])
                q1 = lcm // matr[s, s] 
                q2 = lcm // matr[i, s]
                matr.row_op(i, lambda val, col: q2 * val - q1 * matr[s, col])
                left.row_op(i, lambda val, col: q2 * val - q1 * left[s, col])
                if q2 < 0:
                    matr.row_op(i, lambda val, col: -val)
                    left.row_op(i, lambda val, col: -val)

 
# Iteration of transformation into Smith normal form, modifies edging starting at position [s, s]
def transform_smith(matr: Matrix, left: Matrix, s: int, out='prim', test=False):
    move_least_to_start(matr, left, s)
    print('1st:') if test else None
    ppr(matr, left) if test else None
 
    modify_edging(matr, left, s)
    print('2nd:') if test else None
    ppr(matr, left) if test else None
 
    null_edging(matr, left, s)
    print('3rd:') if test else None
    ppr(matr, left) if test else None

    modify_upper_edging(matr, left, s, out=out)
    print('4th:') if test else None
    ppr(matr, left) if test else None
 
# Checks whether south-western block of matrix, starting at pos [s, s], is zero
def next_block_empty_or_null(matr: Matrix, s: int):
    rows, cols = len(matr.col(0)), len(matr.row(0))
    return matr[s + 1: rows, s + 1: cols].is_zero_matrix
 
 
# Diagonalize input integer mat, using only row transformations, 
# for input mat A, find L * A = D, where D is diagonal, and L is left square matrix
# return: L, D
def gauss_elim(matr: Matrix, out='prim', test = False):
    matr = matr.copy()
    rows, cols = len(matr.col(0)), len(matr.row(0))
    left = Matrix.eye(rows)
 
    if matr.is_zero_matrix:
        return matr, left
 
    for s in range(0, min(rows, cols)):
        transform_smith(matr, left, s, out=out, test=test)
 
        if next_block_empty_or_null(matr, s):
            rank = s + 1
            break

    assert rank == 3, (rank, Matrix)
    return matr, left

def gauss_elim_np(mat, out='prim', test=False):
    assert np.linalg.norm(mat - np.round(mat)) < 1e-6, ('input mat should be integer matrix!', mat)
    SM, L = gauss_elim(Matrix([ [int(round(a)) for a in line] for line in mat ]), out=out, test=test)
    return np.array(SM).astype(int), np.array(L).astype(int)

 
if __name__ == '__main__': 

   #mat = np.array([[-1,1,1], [1, -1, 1], [1, 1,-1]])
   #mat = np.array([[0,1,1], [1, 0, 1], [1, 1,0]])
   #mat = np.array([[1,-1, 0], [1, 1, 0], [0, 0,-1]])
   #mat = np.array([[2, 1, 1], [-1, 1, 1], [-1, -2, 1]]) 
    mat = np.array([[2,0,2],[3,3,0],[0,0,6]])
    mat, left = gauss_elim_np(mat, out='prim', test=1)
    print('prim:',mat, mat/2)
    mat, left = gauss_elim_np(mat, out='conv', test=1)
    print('result:\n', mat, '\nleft:\n', left)
