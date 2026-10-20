import numpy as np
from spglib import *
from numpy.linalg import norm, inv, det
import os
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
import pickle
import warnings
from pymatgen.io.cif import CifParser
from collections import OrderedDict
from math import cos, sin, acos, asin, pi, sqrt, tan
from itertools import product

from .small_func import *
from .PG_utils import sort_rot
from .SG_utils import identify_SG_lattice
from .SG_isomorphism import find_sg_iso_transform

def unique_list(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def change_tau(tau):
    """Return a fractional translation without rounding away rational offsets."""
    return np.mod(np.asarray(tau, dtype=float), 1.0)


def same_translation_mod_lattice(tau1, tau2, tol=1e-4):
    """Compare translations on the 3-torus, including values near 0 and 1."""
    diff = np.asarray(tau1) - np.asarray(tau2)
    return norm(diff - np.rint(diff)) < tol

def sort_array(target):
    """Sort position/moment rows using position keys, preserving moment precision."""
    target = np.asarray(target).copy()
    # Retain the existing coordinate quantization; it must not quantize moments.
    target[:, :3] = np.round(target[:, :3], 5)
    order = np.lexsort((target[:, 2], target[:, 1], target[:, 0]))
    return target[order]

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

    return pos_list

def findGnum(rot_list, trans_list, lattice):
    position = generate_coord(rot_list, trans_list)
    numbers = np.arange(4).tolist() * np.size(rot_list, axis=0)
    cellG= (lattice, position, numbers)
    dataset = get_symmetry_dataset(cellG,symprec=1e-3)
    # print(dataset)
    # print(cellG)
    # print(rot_list)
    # print(trans_list)
    Gnum = int(dataset['number'])
    # print('find Gnum', Gnum)
    transform_matrix = dataset['transformation_matrix']
    shift = dataset['origin_shift']
    shift = change_shift(shift, Gnum)
    Ik = det(transform_matrix)
    # print(Ik,transform_matrix,shift)
    return [Gnum, transform_matrix, shift]

def findQlabel(rot_list):
    rot_list = unique_matrix(rot_list)
    # print(rot_list)
    coord = [[0, 0, 0]]
    elements = ['C']
    elements = ['C']
    gen_pts = [np.array([0.1722, 0.6933, 0.9344]),
                np.array([0.8399, 0.5677, 0.0655]),
                np.array([0.1234, 0.4567, 0.0876]),
                np.array([0.2468, 0.5721, 0.7834])]
    for R in rot_list:
        for p in gen_pts:
           #pos_list.append(latt_home(R @ p + t))
            coord.append((R @ p.T).T)
            elements.append('H')
    mol = Molecule(elements, coord)
    # print(mol)
    point_group = PointGroupAnalyzer(mol, tolerance=0.00001,eigen_tolerance=0.001, matrix_tolerance=0.001)
    sch_label = point_group.sch_symbol
    return sch_label
    
def std_cell(cell): # standard the cell by sort the position on each atom
    def position_and_magmom(positions, magmoms):
        assert np.size(positions) == np.size(magmoms)
        num_atom = np.size(positions, 0)
        pos_mag = np.zeros((num_atom, 6))
        pos_mag[:, 0:3] = positions
        pos_mag[:,3:] = magmoms
        return pos_mag 
    def mod_pos(cell):
        lattice = cell[0].copy()
        position = cell[1].copy()
        numbers = cell[2].copy()
        mag = cell[3].copy()
        pos_mag = position_and_magmom(position, mag)
        num_atom = np.size(position, 0)
        poss = []
        for at in range(num_atom):
            pos = position[at]
            mod_position = [(pos[0]+1e-6)%1, (pos[1]+1e-6)%1, (pos[2]+1e-6)%1]
            poss.append(mod_position)

        poss = np.array(poss)
        cell_new = (lattice, poss, numbers, mag)
        return cell_new
    def get_deg(numbers):
        n = 1
        num_deg = []
        number_ar = np.array(numbers)
        while n in numbers:
            num_deg.append(len(np.nonzero(number_ar == n)[0]))
            n = n + 1
        return num_deg
    cell = mod_pos(cell)
    lattice = cell[0].copy()
    position = cell[1].copy()
    numbers = cell[2].copy()
    mag = cell[3].copy()
    pos_mag = position_and_magmom(position, mag)
    deg_points = get_deg(numbers)
    num_type = int(max(numbers))
    # print(deg_points)
    line = 0
    # numbers is in the form (1,1,1,1,2,2,2,2,3,3,3,3...)
    for num in range(num_type):
        deg = deg_points[num]
        if line + deg < len(numbers):
            array_atom = pos_mag[line : line+deg]
        else:
            array_atom = pos_mag[line :]
        target = array_atom.copy()
        target = sort_array(target)
        #print(target)
        if line + deg < len(numbers):
            pos_mag[line : line+deg] = target
        else:
            pos_mag[line :] = target
        line = line + deg
    positions = pos_mag[:, 0:3]
    magmoms = pos_mag[:,3:]

    cell_standard = (lattice, positions, numbers, magmoms)

    return cell_standard
    
def schoenflies_to_hm(schoenflies_symbol):
    symbol_dict = {
        "C1": "1",
        "Ci": "-1",
        "Cs": "m",
        "C2": "2",
        "C2h": "2/m",
        "D2": "222",
        "C2v": "mm2",
        "C3": "3",
        "C3i": "-3",
        "D3": "32",
        "C3v": "3m",
        "D3d": "-3m",
        "C4": "4",
        "S4": "-4",
        "C4h": "4/m",
        "D4h": "4/mmm",
        "D4": "422",
        "C4v": "4mm",
        "D2d": "-42m",
        "D2h": "mmm",
        "C6": "6",
        "C3h": "-6",
        "C6h": "6/m",
        "D6": "622",
        "C6v": "6mm",
        "D3h": "-6m2",
        "D6h": "6/mmm",
        "T": "23",
        "Th": "m-3",
        "Td": "-43m",
        "O": "432",
        "Oh": "m-3m",
        "S6": "-3",
        "C12v": "C12v"
    }
    
    return symbol_dict.get(schoenflies_symbol, "")

def read_poscar(file_name = 'POSCAR'): #read the POSCAR to get the cell
    lattice  = np.zeros((3, 3))
    deg_of_points = []
    numbers = []
    mag = []
    # file_name = 'POSCAR_test'
    with open(file_name, 'r') as f:
        f_lines = f.readlines()
        for cnt, line in enumerate(f_lines):
            # cnt = 0,1 are meaningless
            # cnt = 2,3,4 show the basis vectors
            if cnt in [2, 3, 4]:
                line = line.strip().split()
                lattice[cnt-2, :] = [float(c) for c in line]
            if cnt  == 6:
                line = line.strip().split()
                deg_of_points = [int(c) for c in line]
                num_of_points = sum(deg_of_points)
                num_of_types = np.size(deg_of_points)
                # get the numbers of cell
                for m in range(num_of_types):
                    for n in range(deg_of_points[m]):
                        numbers.append(m+1)
                positions = np.zeros((num_of_points, 3))
            if cnt > 7 and cnt < (7 + num_of_points + 1):
                line = line.strip().split()
                if len(line) == 3:
                    positions[cnt-8, :] = [float(c) for c in line]
                    mag.append([0, 0, 0])
                elif len(line) == 6:
                    positions[cnt-8, :] = [float(c) for c in line[:3]]
                    mag.append([float(c) for c in line[3:]])
                else:
                    # print('line', cnt+1 ,'not valid POSCAR input !!!')
                    raise ValueError('Invalid input, please check POSCAR line ' + str(cnt+1) )
        cell = (lattice, positions, numbers, np.array(mag)) 
    return cell

# def get_cell(file, mag):
#     cell = read_poscar(file)
#     # dataset = get_symmetry_dataset(cell)
#     # print(dataset['number'])
#     # mag = np.array([[1.5, 2.598, 0], [-3, 0, 0], [-3, 0, 0], [1.5, 2.598, 0], [1.5, -2.598, 0], [1.5, -2.598, 0], [0 ,0, 0], [0, 0, 0]])
#     [lattice, positions, numbers] = [cell[0], cell[1], cell[2]]
#     cell_mag = (lattice, positions, numbers, mag)
#     return cell_mag

def identity_vec(v1 ,v2, tol = 1e-4):
    diff = np.linalg.norm(v1 - v2)
    if diff < tol:
        return True
    else:
        return False

def findH(cell, tol, mag_tol=1e-4): # find the pure lattice space group number
    lattice = cell[0].copy()
    position = cell[1].copy()
    numbers = cell[2].copy()
    mag = cell[3].copy()
    num_atom = np.size(numbers, 0)
    # print('num of atom', num_atom)
    new_numbers = []
    numbers_bascket = []
    mag_bascket = []
    num_list = []
    num = 1
    for at in range(num_atom):
        flag = True
        if at == 0:
            numbers_bascket.append(numbers[at])
            mag_bascket.append(mag[at])
            new_numbers.append(1)
            num_list.append(1)
            num = num + 1
        else:
            num_bascket = np.size(numbers_bascket)
            for i in range(num_bascket):
                if numbers[at] == numbers_bascket[i] and identity_vec(mag[at], mag_bascket[i], mag_tol):
                    num_need = num_list[i]
                    new_numbers.append(num_need)
                    flag = False
                    break    
            if flag:
                numbers_bascket.append(numbers[at])
                mag_bascket.append(mag[at])
                new_numbers.append(num)
                num_list.append(num)
                num = num + 1

    cellH = (lattice, position, new_numbers)
    H_dataset = get_symmetry_dataset(cellH, symprec=tol)
    H_number = H_dataset['number']
    Hop = get_symmetry(cellH, symprec=tol)
    Hrot = H_dataset['rotations']
    Htran = H_dataset['translations']
    # print('H',H_number,Hrot,Htran)
    pos_gen = generate_coord(Hrot, Htran)
    numbers = np.arange(4).tolist() * np.size(Hrot, axis=0)
    cellH_new= (lattice, pos_gen, numbers)
    # print(cellH_new)
    H_dataset = get_symmetry_dataset(cellH_new)
    a = H_dataset
    # print(H_dataset)
    H_number = H_dataset['number']
    return [H_number, Hrot, Htran]

def findG(cell, tol): # non-magnetic spacce group, return the list of rotations and translations [0]Gnum [1]rots [2]trans
    lattice = cell[0].copy()
    position = cell[1].copy()
    numbers = cell[2].copy()
    cellG = (lattice, position, numbers)
    Gnum = get_spacegroup(cellG)
    # print(Gnum)
    Gop = get_symmetry(cellG, symprec=tol)
    Grot = Gop['rotations']
    Gtran = Gop['translations']
    return [Gnum, Grot, Gtran]

def findQ(cell, mag_tol=1e-4): # largest possible spin point group
    dim = findDimension(cell, mag_tol)
    if dim == 1:
        # Representatives modulo the continuous spin-only stabilizer.
        return [np.eye(3), -np.eye(3)]
    if dim == 0:
        raise ValueError('Nonmagnetic structure: use the ordinary space group')
    # if dim == 0:
    #     return '1'
    # if dim == 1:
    #     return 'm'
    mag = cell[3].copy()
    numbers = cell[2].copy()
    elements = []
    coord = []
    num = np.size(numbers, 0)
    for at in range(num):
        if not identity_vec(mag[at], [0, 0, 0], mag_tol):
            coord.append(mag[at])
    # append [0, 0, 0] to the mass center
    coord.append(np.array([0, 0, 0]))
    coord = np.unique(coord, axis = 0)
    num_uni = np.size(coord, axis = 0)
    for i in range(num_uni):
        elements.append('H')

    # num2ele = {1: 'H', 2:'He', 3:'Li', 4:'Be', 5:'B', 6: 'C', 7:'N', 8:'O', 9:'F'}
    # for num in new_numbers:
    #     elements.append(num2ele[num])
    mol = Molecule(elements, coord)
    # print(elements, coord)
    point_group = PointGroupAnalyzer(mol, tolerance=mag_tol, eigen_tolerance=0.0001, matrix_tolerance=0.00001)
    # print('posible Q')
    # print(point_group.sch_symbol)
    op = point_group.get_symmetry_operations()
    op_list = []
    for i in op:
        op_list.append(i.rotation_matrix) # rotation operation list
    label = point_group.sch_symbol
    # print(op_list)
    return op_list
        
def _positive_tolerance(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def findDimension(cell, mag_tol=1e-4, return_details=False):
    """Resolve magnetic dimension using an absolute per-atom moment tolerance.

    For a line or plane, require every residual from the SVD subspace to be
    below mag_tol / 2. Reflection of that residual then costs at most mag_tol,
    consistent with the operation check. No input moment is rounded/projected.
    Unlike a singular-value cutoff this criterion does not grow with atom count.
    """
    mag_tol = _positive_tolerance(mag_tol, 'mag_tol')
    mag = np.asarray(cell[3], dtype=float)
    if mag.ndim != 2 or mag.shape[1] != 3 or not len(mag) or not np.isfinite(mag).all():
        raise ValueError('magnetic moments must be a nonempty finite (N, 3) array')
    _, singular_values, axes = np.linalg.svd(mag, full_matrices=False)
    residuals = [float(np.max(norm(mag, axis=1)))]
    for rank in (1, 2):
        basis = axes[:rank]
        residuals.append(float(np.max(norm(mag - (mag @ basis.T) @ basis, axis=1))))
    if residuals[0] < mag_tol:
        dimension = 0
    elif residuals[1] < mag_tol / 2:
        dimension = 1
    elif residuals[2] < mag_tol / 2:
        dimension = 2
    else:
        dimension = 3
    details = dict(dimension=dimension, mag_tol=mag_tol,
                   subspace_tolerance=mag_tol / 2,
                   singular_values=singular_values.tolist(),
                   max_residual_by_rank=residuals)
    return details if return_details else dimension

def mcif2cell(file_name):
    cc = CifParser(file_name)
    structure = cc.get_structures(primitive=False)[0]
    atom_types = [site.species_string for site in structure]
    atom_map = list(OrderedDict.fromkeys(atom_types))
    element_map = {element: i+1 for i, element in enumerate(atom_map)}
    numbers = [element_map[site.species_string] for site in structure]

    # numbers = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    # print(numbers)
    lattice = structure.lattice.matrix
    position = structure.frac_coords
    m = structure.site_properties['magmom']
    mag = []
    for i in m:
        mag.append(i.moment)
    cell = (lattice, position, numbers, mag)
    return cell

def _ssg_candidates(ssg_list, operations, dim):
    """Apply the database's coarse invariants before comparing operations."""
    search = f"{operations['Gnum']}.{operations['Ik']}.{operations['It']}"
    return [
        record for record in ssg_list
        if record['dim'] == dim
        and record['search'] == search
        and record['Hid'] == int(operations['Hnum'])
        and record['eqvPg'] == str(operations['QLabel'])
    ]


def _affine_key(rotation, shift):
    """Deduplicate an exact affine setting.

    A primitive translation of G can act differently on G/H when H has a
    supercell, so integer shifts cannot be reduced modulo the G lattice here.
    """
    rotation_key = tuple(np.rint(np.asarray(rotation).ravel() * 1e7).astype(int))
    shift_key = tuple(np.rint(np.asarray(shift) * 1e7).astype(int))
    return rotation_key, shift_key


def _automorphism_transforms(operations, iso_data, offsets):
    """Compose material-to-standard setting with each automorphism of G.

    If x_std = P x_material + p and the automorphism is x' = A x_std + a,
    the combined setting is x' = (A P) x_material + (A p + a).
    """
    base_rotation = np.asarray(operations['transformation_matrix'])
    base_shift = np.asarray(operations['original_shift'])
    identity = (np.eye(3), [np.zeros(3)])
    automorphisms = [identity] + list(zip(iso_data['iso_rot'], iso_data['iso_tau']))
    for offset in offsets:
        shifted_origin = base_shift + np.asarray(offset)
        for rotation, shifts in automorphisms:
            for shift in shifts:
                yield rotation @ base_rotation, rotation @ shifted_origin + shift


def _quarter_origin_offsets():
    """Finite extra origin shifts used in the paper's identification procedure."""
    for i, j, k in product(range(4), repeat=3):
        if i or j or k:
            yield np.array([i, j, k], dtype=float) / 4


def _translation_cosets(candidate, gid):
    """Represent T_G / T_H in the conventional coordinates of G."""
    supercell = np.rint(np.asarray(candidate['superCell'])).astype(int)
    index = round(abs(np.linalg.det(supercell)))
    if index == 1:
        return [np.zeros(3)]
    inverse = np.linalg.inv(supercell)
    representatives = []
    for coordinates in product(range(index), repeat=3):
        vector = np.asarray(coordinates, dtype=int)
        if any(np.allclose(inverse @ (vector - previous),
                           np.rint(inverse @ (vector - previous)), atol=1e-8)
               for previous in representatives):
            continue
        representatives.append(vector)
        if len(representatives) == index:
            break
    if len(representatives) != index:
        raise ValueError(f'Could not enumerate translation cosets for {candidate["ssgNum"]}')
    primitive_basis = identify_SG_lattice(gid)[1]
    return [primitive_basis @ vector for vector in representatives]


def _triclinic_index_two_setting(candidate, operations):
    """Map an index-two H translation lattice for G=P-1 using GL(3,Z).

    Such sublattices are kernels of a nonzero parity character on Z^3.
    Construct a unimodular matrix by elementary column operations instead
    of enumerating the infinite affine normalizer of P-1.
    """
    if operations['Gnum'] != 2 or operations['Ik'] != 2:
        return None

    def parity_normal(generators):
        rounded = np.rint(generators)
        if not np.allclose(generators, rounded, atol=1e-5):
            return None
        vectors = rounded.astype(int)
        normals = [np.asarray(bits, dtype=int)
                   for bits in product((0, 1), repeat=3)
                   if any(bits) and all(np.dot(bits, v) % 2 == 0 for v in vectors)]
        return normals[0] if len(normals) == 1 else None

    P = np.asarray(operations['transformation_matrix'])
    material_generators = [P[:, i] for i in range(3)]
    material_generators += [P @ np.asarray(t) for r, t in
                            zip(operations['HRotC'], operations['HTauC'])
                            if norm(np.asarray(r) - np.eye(3)) < 1e-5]
    material_normal = parity_normal(np.asarray(material_generators))
    database_normal = parity_normal(np.asarray(candidate['superCell']).T)
    if material_normal is None or database_normal is None:
        return None

    A = np.eye(3, dtype=int)
    row = database_normal.copy()

    def swap_columns(i, j):
        nonlocal A, row
        elementary = np.eye(3, dtype=int)
        elementary[:, [i, j]] = elementary[:, [j, i]]
        A = A @ elementary
        row = (row @ elementary) % 2

    def add_column(source, target):
        nonlocal A, row
        elementary = np.eye(3, dtype=int)
        elementary[source, target] = 1
        A = A @ elementary
        row = (row @ elementary) % 2

    pivot = int(np.flatnonzero(row)[0])
    swap_columns(0, pivot)
    for j in (1, 2):
        if row[j]:
            add_column(0, j)
    pivot = int(np.flatnonzero(material_normal)[0])
    swap_columns(0, pivot)
    for j in range(3):
        if j != pivot and material_normal[j]:
            add_column(pivot, j)
    assert np.array_equal(row, material_normal)
    return A


def search4ssg(cell, ssg_list, tol=1e-4, operations=None, return_details=False,
               *, mag_tol=1e-4, translation_tol=1e-4, rotation_tol=1e-3, spin_tol=1e-2):
    """Match found operations to one SSG database number.

    ``operations`` lets callers reuse the first-stage result instead of
    repeating the expensive operation search. ``return_details`` exposes the
    matching stage and counts for debugging while the default keeps the
    original string return type.
    """
    tol = _positive_tolerance(tol, 'tol')
    mag_tol = _positive_tolerance(mag_tol, 'mag_tol')
    translation_tol = _positive_tolerance(translation_tol, 'translation_tol')
    rotation_tol = _positive_tolerance(rotation_tol, 'rotation_tol')
    spin_tol = _positive_tolerance(spin_tol, 'spin_tol')
    dimension_details = findDimension(cell, mag_tol, return_details=True)
    dim = dimension_details['dimension']
    if dim == 0:
        raise ValueError('Nonmagnetic structure: use the ordinary space group')
    if operations is not None:
        for key, expected in (('mag_tol', mag_tol), ('symprec', tol), ('dimension', dim)):
            if key in operations and operations[key] != expected:
                raise ValueError(f'Provided operations have different {key}; recompute with matching tolerances')
    operations = operations if operations is not None else findAllOp(cell, tol, mag_tol)
    candidates = _ssg_candidates(ssg_list, operations, dim)
    used_fallback = False
    if not candidates:
        operations = findAllOp_v2(cell, tol, mag_tol)
        candidates = _ssg_candidates(ssg_list, operations, dim)
        used_fallback = True

    details = {
        'dimension_details': dimension_details,
        'tolerances': dict(symprec=tol, mag_tol=mag_tol, translation_tol=translation_tol,
                           rotation_tol=rotation_tol, spin_tol=spin_tol),
        'Gnum': operations['Gnum'],
        'Hnum': operations['Hnum'],
        'Ik': operations['Ik'],
        'It': operations['It'],
        'QLabel': operations['QLabel'],
        'candidate_numbers': [record['ssgNum'] for record in candidates],
        'used_fallback_operations': used_fallback,
        'transforms_tested': 0,
        'h_matches': 0,
        'q_matches': 0,
    }

    def finish(status, numbers=(), stage=None):
        details['status'] = status
        details['matching_numbers'] = sorted(numbers)
        details['stage'] = stage
        details['number'] = next(iter(numbers)) if status == 'matched' else None
        if return_details:
            return details
        return details['number'] if status == 'matched' else 'need more loop'

    if not candidates:
        return finish('no_candidates')

    seen = set()

    def try_transforms(transforms, stage):
        matches = set()
        for transform, shift in transforms:
            key = _affine_key(transform, shift)
            if key in seen:
                continue
            seen.add(key)
            details['transforms_tested'] += 1
            for candidate in candidates:
                number = candidate['ssgNum']
                if number in matches:
                    continue
                if not checkH(candidate, operations, shift, transform, translation_tol, rotation_tol):
                    continue
                details['h_matches'] += 1
                if checkQ(candidate, operations, dim, shift, transform, translation_tol, rotation_tol, spin_tol):
                    details['q_matches'] += 1
                    matches.add(number)
            if len(matches) > 1:
                return finish('ambiguous', matches, stage)
        if len(matches) == 1:
            return finish('matched', matches, stage)
        return None

    base_rotation = np.asarray(operations['transformation_matrix'])
    base_shift = np.asarray(operations['original_shift'])
    result = try_transforms([(base_rotation, base_shift)], 'spglib')
    if result is not None:
        return result

    # Cheap origin shifts should precede the potentially unbounded GL(3, Z)
    # automorphism search for triclinic and monoclinic space groups.
    lattice_shifts = []
    for candidate in candidates:
        lattice_shifts.extend(_translation_cosets(candidate, operations['Gnum'])[1:])

    def local_origin_settings():
        # Eighth-cell shifts are required by some cubic settings (e.g. SG 201
        # and 203).  Search them before the more expensive G automorphisms.
        shifts = [np.zeros(3)] + lattice_shifts
        for coordinates in product(range(8), repeat=3):
            if coordinates == (0, 0, 0):
                continue
            offset = np.asarray(coordinates, dtype=float) / 8
            for lattice_shift in shifts:
                yield base_rotation, base_shift + offset + lattice_shift

    result = try_transforms(local_origin_settings(), 'extra_origin_shift')
    if result is not None:
        return result

    if len(candidates) == 1:
        triclinic = _triclinic_index_two_setting(candidates[0], operations)
        if triclinic is not None:
            def aligned_triclinic_settings():
                transform = triclinic @ base_rotation
                shift = triclinic @ base_shift
                for coordinates in product(range(8), repeat=3):
                    offset = np.asarray(coordinates, dtype=float) / 8
                    for lattice_shift in [np.zeros(3)] + lattice_shifts:
                        yield transform, shift + offset + lattice_shift

            result = try_transforms(aligned_triclinic_settings(),
                                    'triclinic_index_two')
            if result is not None:
                return result

    iso_data = find_sg_iso_transform(operations['Gnum'])
    result = try_transforms(
        _automorphism_transforms(operations, iso_data, [np.zeros(3)]),
        'automorphism',
    )
    if result is not None:
        return result

    def with_lattice_shifts(offsets, include_zero=False):
        shifts = ([np.zeros(3)] if include_zero else []) + lattice_shifts
        for transform, shift in _automorphism_transforms(operations, iso_data, offsets):
            for lattice_shift in shifts:
                yield transform, shift + lattice_shift

    if lattice_shifts:
        result = try_transforms(with_lattice_shifts([np.zeros(3)]),
                                'translation_coset')
        if result is not None:
            return result

    result = try_transforms(
        with_lattice_shifts(_quarter_origin_offsets(), include_zero=True),
        'extra_origin_shift',
    )
    if result is not None:
        return result
    return finish('no_equivalent_operations', stage='extra_origin_shift')


def identity_cell(cell1, cell2, mag_tol=1e-4): 
    cell1_std = std_cell(cell1)
    cell2_std = std_cell(cell2)
    # print(cell1_std)
    # print(cell2_std)
    # position must be the same since the space group is true
    # only need to justify the magnetic identity
    mag1 = cell1_std[3].copy()
    mag2 = cell2_std[3].copy()
    for cnt, m1 in enumerate(mag1):
        m2 = mag2[cnt]
        if not identity_vec(m1, m2, mag_tol):
            return False
    return True

def checkQ(ss, ssg_dict, dim, shift, transform, translation_tol=1e-4, rotation_tol=1e-3, spin_tol=1e-2):
    def check_rot(rot1, rot2, dim):
        if dim == 1:
            if abs(det(rot1) - det(rot2)) < spin_tol:
                return True
            else:
                return False
        # angle1, axis1, det1 = get_rotation(rot1)
        # angle2, axis2, det2 = get_rotation(rot2)
        # if abs(angle1-angle2) < 1e-1 and abs(det1-det2) < 1e-1:
        #     return True
        # else:
        #     return False
        diff_trace = abs(np.trace(rot1) - np.trace(rot2))
        if diff_trace < spin_tol:
            return True
        else:
            return False
    QRotC_std = ss['QRotC']
    QTauC_std = ss['QTauC']
    URot_std_list = ss['URot']
    QRotC = []
    QTauC = []
    # print('check q of ', ss['ssgNum'])
    # trnasform = ssg_dict['transformation_matrix']
    transform_inv = np.linalg.inv(transform)
    # shift = ssg_dict['original_shift']
    Rot_list = ssg_dict['RotC']
    Tau_list = ssg_dict['TauC']
    spin_list = ssg_dict['spin']
    # print('material q: rot, tau and spin')
    # print_matlist(Rot_list)
    # print_matlist(Tau_list)
    # print_matlist(spin_list)
    for cnt, rot in enumerate(QRotC_std):
        tau = QTauC_std[cnt]
        QRotC.append(transform_inv@rot@transform)
        QTauC.append(transform_inv@(rot@shift+tau-shift))
    # print('standard lattice operations of q after transformation')
    # print_matlist(QRotC)
    # print_matlist(QTauC)
    # flag_rep = False
    rep_T = 0
    for URot_std in URot_std_list:
        flag2 = True
        for cnt1, rot1 in enumerate(QRotC):
            flag = False
            # flag2 = True
            for cnt2, rot2 in enumerate(Rot_list):
                t1 = QTauC[cnt1]
                t2 = Tau_list[cnt2]
                if same_translation_mod_lattice(t1, t2, translation_tol) and norm(rot1-rot2) < rotation_tol:
                    flag = True
                    spin_rot1 = spin_list[cnt2]
                    spin_rot2 = URot_std[cnt1]
                    spin_rot1 = np.array(spin_rot1)
                    spin_rot2 = np.array(spin_rot2)
                    if not check_rot(spin_rot1, spin_rot2, dim):
                        # print('spin part not identity')
                        flag2 = False
                        # print(t1,rot1,spin_rot1, spin_rot2)
                        # print(rot2,Tau_list[cnt2], spin_rot1,spin_rot2)
            if not flag:
                # print(Rot_list, Tau_list, spin_list)
                # print(QRotC, QTauC)
                # print('some space operations not found')
                return False
        if flag2:
            # print('rep OK')
            rep_T = rep_T + 1
        # else:
            # print('spin part not identity')
    if rep_T == 1:
        return True
    if rep_T > 1:
        # print('ohhh, more than one rep ture')
        return True
    return False

def checkH(ss,ssg_dict, shift, transform, translation_tol=1e-4, rotation_tol=1e-3):
    gid = ssg_dict['Gnum']
    HRotC_std = list(ss['HRotC'])
    HTauC_std = list(ss['HTauC'])
    # trnasform = ssg_dict['transformation_matrix']
    tau_col = ss['superCell']
    t1 = np.array([tau_col[0][0], tau_col[1][0], tau_col[2][0]]) 
    t2 = np.array([tau_col[0][1], tau_col[1][1], tau_col[2][1]]) 
    t3 = np.array([tau_col[0][2], tau_col[1][2], tau_col[2][2]])
    # prim_vec = np.array([[2. / 3, -1. / 3, -1. / 3], 
    #                     [1. / 3, -2. / 3, 1. / 3], 
    #                     [1. / 3, 1 / 3, 1. / 3]])
    # prim_vec = np.array([[2. / 3, 1. / 3, 1. / 3], 
    #                       [-1. / 3, 1. / 3, 1. / 3], 
    #                       [-1. / 3, -2. / 3, 1. / 3]]).T  
    prim_vec = identify_SG_lattice(gid)[1] # each col is a prim basis vector               
    for t_append in [t1, t2, t3]:
        HRotC_std.append(np.eye(3))
        HTauC_std.append(prim_vec @ t_append)
    transform_inv = np.linalg.inv(transform)
    # shift = ssg_dict['original_shift']
    Rot_list = ssg_dict['HRotC']
    Tau_list = ssg_dict['HTauC']
    HRotC = []
    HTauC = []
    for cnt, rot in enumerate(HRotC_std):
        tau = HTauC_std[cnt]
        HRotC.append(transform_inv@rot@transform)
        # print(transform_inv,rot,shift,tau,shift)
        HTauC.append(transform_inv@(rot@shift+tau-shift))
    # print(trnasform, shift)
    # print(Rot_list, Tau_list)
    # print(HRotC, HTauC)
    for cnt1, rot1 in enumerate(HRotC):
        flag = False
        for cnt2, rot2 in enumerate(Rot_list):
            t1 = HTauC[cnt1]
            t2 = Tau_list[cnt2]
            if same_translation_mod_lattice(t1, t2, translation_tol) and norm(rot1-rot2) < rotation_tol:
                flag = True
        if not flag:
            # print(Rot_list, Tau_list, spin_list)
            # print(QRotC, QTauC)
            return False
    return True

def unique_matrix(matrix_list, tol=1e-3):
    unique_list = [matrix_list[0]]  # add the first
    for mat in matrix_list[1:]:
        is_unique = True
        for uni in unique_list:
            if np.linalg.norm(mat - uni) < tol:
                is_unique = False
                break
        if is_unique:
            unique_list.append(mat)
    return unique_list

def twoD2threeD(cell, mag_tol=1e-4):
    """Lift the spin plane along its SVD normal, avoiding tiny cross products."""
    if findDimension(cell, mag_tol) != 2:
        raise ValueError('twoD2threeD requires a planar magnetic structure')
    mag = np.asarray(cell[3], dtype=float)
    _, _, axes = np.linalg.svd(mag, full_matrices=False)
    normal = np.cross(axes[0], axes[1]) * np.max(norm(mag, axis=1))
    return (cell[0].copy(), cell[1].copy(), cell[2].copy(), mag + normal)


def findAllOp(cell, tol, mag_tol=1e-4):
    tol = _positive_tolerance(tol, 'tol')
    mag_tol = _positive_tolerance(mag_tol, 'mag_tol')
    dim = findDimension(cell, mag_tol)
    if dim == 0:
        raise ValueError('Nonmagnetic structure: use the ordinary space group')
    # id dim = 2, transfer it into a 3d which will maintain all the operation, but still .P
    if dim == 2:
        cell = twoD2threeD(cell, mag_tol)
    lattice = cell[0].copy()
    position = cell[1].copy()
    numbers = cell[2].copy()
    mag = cell[3].copy()
    out_spin = []
    out_rot = []
    out_tran = []
    spinOp = findQ(cell, mag_tol)
    latticeOp = findG(cell, tol)
    non_mag_G = latticeOp[0]
    rot = latticeOp[1]
    tran = latticeOp[2]
    assert np.size(rot, 0) == np.size(tran, 0)
    sg_op_num = np.size(rot, 0)
    for i in range(sg_op_num):
        r = rot[i]
        t = tran[i]
        for spin_rot in spinOp:
            pos_new = []
            mag_new = []
            for pos in position:
                pos_new.append((r @ pos.T + t).T)
            for m in mag:
                mag_new.append((spin_rot @ m.T).T)
            # print(t)
            # print(spin_rot)
            cell_new = (lattice, pos_new, numbers, mag_new)
            if identity_cell(cell, cell_new, mag_tol):
                out_spin.append(spin_rot)
                out_rot.append(r)
                out_tran.append(t)
    # using the dimension to identify the ssg
    # print(out_spin)
    # print(out_rot)
    # print(out_tran)
    fH = findH(cell, 1e-2, mag_tol)
    # fH = findH_v2(out_spin, out_rot, out_tran, cell)
    Hnum = fH[0]
    Hrot = fH[1]
    Htran = fH[2]
    H_op_num = np.size(Hrot, axis=0)
    G_op_num = np.size(out_rot, axis=0)
    H_point_num = np.size(unique_matrix(Hrot), axis=0)
    G_point_num = np.size(unique_matrix(out_rot), axis=0)
    # print(H_point_num, G_point_num)
    # get G number
    fG = findGnum(out_rot, out_tran, lattice)
    Gnum = fG[0]
    transform = fG[1]
    shift = fG[2]
    # get ik and it
    It = int(round(G_point_num/H_point_num))
    IKIT = G_op_num/H_op_num
    assert abs(IKIT - round(IKIT)) < 1e-3
    IKIT = round(IKIT)
    Ik = IKIT/It
    assert abs(Ik - round(Ik)) < 1e-3
    Ik = round(Ik)
    # find Q
    # print(out_spin)
    spin_uniqe = unique_matrix(out_spin)
    # print(spin_uniqe)
    # when dim == 1, it can only be fermmomagnetic or anti-ferromagnetic
    if dim == 1:
        if np.size(spin_uniqe, axis=0) == 1:
            Gnum = Hnum
            assert It == 1
            Ik = 1
            spin_label = 'C1'
        else:
            assert np.size(spin_uniqe, axis=0) == 2
            assert It == 1 or 2
            Ik = int(2/It)
            spin_label = 'Cs'

    else:
        spin_label = findQlabel(out_spin)
        # print(spin_label)

    return {'dimension': dim, 'mag_tol': mag_tol, 'symprec': tol, 'spin': out_spin, 'It': It, 'Hnum': Hnum, 'Ik': Ik, 'Gnum': Gnum, 'QLabel': schoenflies_to_hm(spin_label), 'RotC': out_rot,
            'TauC': out_tran, 'spin': out_spin, 'transformation_matrix': transform, 'original_shift': shift, 'HRotC': Hrot, 'HTauC': Htran}

def findAllOp_v2(cell, tol, mag_tol=1e-4):
    tol = _positive_tolerance(tol, 'tol')
    mag_tol = _positive_tolerance(mag_tol, 'mag_tol')
    dim = findDimension(cell, mag_tol)
    if dim == 0:
        raise ValueError('Nonmagnetic structure: use the ordinary space group')
    # id dim = 2, transfer it into a 3d which will maintain all the operation, but still .P
    if dim == 2:
        cell = twoD2threeD(cell, mag_tol)
    lattice = cell[0].copy()
    position = cell[1].copy()
    numbers = cell[2].copy()
    mag = cell[3].copy()
    out_spin = []
    out_rot = []
    out_tran = []
    spinOp = findQ(cell, mag_tol)
    latticeOp = findG(cell, tol)
    non_mag_G = latticeOp[0]
    rot = latticeOp[1]
    tran = latticeOp[2]
    assert np.size(rot, 0) == np.size(tran, 0)
    sg_op_num = np.size(rot, 0)
    for i in range(sg_op_num):
        r = rot[i]
        t = tran[i]
        for spin_rot in spinOp:
            pos_new = []
            mag_new = []
            for pos in position:
                pos_new.append((r @ pos.T + t).T)
            for m in mag:
                mag_new.append((spin_rot @ m.T).T)
            # print(t)
            # print(spin_rot)
            cell_new = (lattice, pos_new, numbers, mag_new)
            if identity_cell(cell, cell_new, mag_tol):
                out_spin.append(spin_rot)
                out_rot.append(r)
                out_tran.append(t)
    # using the dimension to identify the ssg
    # print(out_spin)
    # print(out_rot)
    # print(out_tran)
    # fH = findH(cell, 1e-6)
    fH = findH_v2(out_spin, out_rot, out_tran, cell)
    Hnum = fH[0]
    Hrot = fH[1]
    Htran = fH[2]
    H_op_num = np.size(Hrot, axis=0)
    G_op_num = np.size(out_rot, axis=0)
    H_point_num = np.size(unique_matrix(Hrot), axis=0)
    G_point_num = np.size(unique_matrix(out_rot), axis=0)
    # print(H_point_num, G_point_num)
    # get G number
    fG = findGnum(out_rot, out_tran, lattice)
    Gnum = fG[0]
    transform = fG[1]
    shift = fG[2]
    # get ik and it
    It = int(round(G_point_num/H_point_num))
    IKIT = G_op_num/H_op_num
    assert abs(IKIT - round(IKIT)) < 1e-3
    IKIT = round(IKIT)
    Ik = IKIT/It
    assert abs(Ik - round(Ik)) < 1e-3
    Ik = round(Ik)
    # find Q
    # print(out_spin)
    spin_uniqe = unique_matrix(out_spin)
    # print(spin_uniqe)
    # when dim == 1, it can only be fermmomagnetic or anti-ferromagnetic
    if dim == 1:
        if np.size(spin_uniqe, axis=0) == 1:
            Gnum = Hnum
            assert It == 1
            Ik = 1
            spin_label = 'C1'
        else:
            assert np.size(spin_uniqe, axis=0) == 2
            assert It == 1 or 2
            Ik = int(2/It)
            spin_label = 'Cs'

    else:
        spin_label = findQlabel(out_spin)
        # print(spin_label)

    return {'dimension': dim, 'mag_tol': mag_tol, 'symprec': tol, 'spin': out_spin, 'It': It, 'Hnum': Hnum, 'Ik': Ik, 'Gnum': Gnum, 'QLabel': schoenflies_to_hm(spin_label), 'RotC': out_rot,
            'TauC': out_tran, 'spin': out_spin, 'transformation_matrix': transform, 'original_shift': shift, 'HRotC': Hrot, 'HTauC': Htran}


def get_rotation(R, return_list=False):
    def round_vec(vec, tol=1e-6):
        return np.array([round(v) if abs(v - np.round(v)) < tol else v for v in vec]) 
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
    if axis[2] != 0:
        if axis[2] < 0:
            angle = 2*pi-angle
        axis = [axis[0]/axis[2], axis[1]/axis[2], 1]
        axis = round_vec(axis)     
    elif axis[1] != 0:
        if axis[1] < 0:
            angle = 2*pi-angle
        axis = [axis[0]/axis[1], 1, 0]
        axis = round_vec(axis)
    else:
        if axis[0] < 0:
            angle = 2*pi-angle
        axis = [1, 0, 0]     
    angle = angle / pi * 180
    if return_list:
        return [det, angle, axis[0], axis[1], axis[2]]
    else:
        return angle, axis, det


def change_shift(shift_orignal, sg): #change the std of spglib to the std of biubo
    shift = np.zeros(3)
    if sg in [48, 86, 126, 210]:
        shift = np.array([-0.25, -0.25, -0.25])
    elif sg in [70, 201]:
        shift = np.array([-0.375, -0.375, -0.375])
    elif sg in [85, 129, 130]:
        shift = np.array([0.25, -0.25, 0])
    elif sg in [50, 59, 125]:
        shift = np.array([-0.25, -0.25, 0])
    elif sg in [133, 134, 137]:
        shift = np.array([0.25, -0.25, 0.25])
    elif sg in [141, 142]:
        shift = np.array([0.5, 0.25, 0.125])
    elif sg == 68:
        shift = np.array([0, -0.25, 0.25])
    elif sg == 88:
        shift = np.array([0, 0.25, 0.125])
    elif sg == 138:
        shift = np.array([0.25, -0.25, -0.25])
    elif sg in [222, 224]:
        shift = np.array([0.25, 0.25, 0.25])
    elif sg == 227:
        shift = np.array([0.125, 0.125, 0.125])
    else:
        shift = 0
    out = shift_orignal - shift
    return out



def checkSuperCell(ss,ssg_dict):
    print('check:', ss['ssgNum'])
    supercell = np.array(ss['superCell'])
    Rot_list = ssg_dict['HRotC']
    Tau_list = ssg_dict['HTauC']
    # spin = ssg_dict['spin']
    flag_t = True
    # find the tau with spin = 1 and rot = 1:
    for cnt, rot in enumerate(Rot_list):
        if not identity_vec(Tau_list[cnt], np.array([0, 0, 0])) and norm(rot - np.eye(3)) < 1e-3:
            tau_check =  Tau_list[cnt]
            flag_t = False
    if flag_t:
        return False
    transform_inv = inv(ssg_dict['transformation_matrix'])
    tau_col = transform_inv @ supercell
    t1 = np.array([tau_col[0][0], tau_col[1][0], tau_col[2][0]]) 
    t2 = np.array([tau_col[0][1], tau_col[1][1], tau_col[2][1]]) 
    t3 = np.array([tau_col[0][2], tau_col[1][2], tau_col[2][2]])
    print(t1, t2, t3)
    print(tau_check)
    if any([identity_vec(t1, tau_check), identity_vec(t2, tau_check), identity_vec(t3, tau_check)]):
        return True
    return False

def identify_PG_from_gid(gid):
    # identify PG label for a given sg
    # return: PG_label
    SG_list = [[1], [2], [3,4,5], [6,7,8,9], np.arange(10,16), np.arange(16,25), np.arange(25,47), np.arange(47,75), 
               np.arange(75,81), [81,82], np.arange(83,89), np.arange(89,99), np.arange(99,111), np.arange(111,123), np.arange(123,143),
               np.arange(143,147), [147,148], np.arange(149,156), np.arange(156,162), np.arange(162,168), 
               np.arange(168,174), [174], [175,176], np.arange(177,183), np.arange(183,187), np.arange(187,191), np.arange(191,195),
               np.arange(195,200), np.arange(200,207), np.arange(207,215), np.arange(215,221), np.arange(221,231)]

    index = [lis for i, lis in enumerate(SG_list) if gid in lis][0]
    return index[0]

def findH_v2(out_spin, rot, tau, cell):
    lattice = cell[0].copy()
    rot_list = []
    tau_list = []
    for cnt, spin in enumerate(out_spin):
        if norm(spin - np.eye(3)) < 1e-3:
            rot_list.append(rot[cnt])
            tau_list.append(tau[cnt])
    position = generate_coord(rot_list, tau_list)
    numbers = np.arange(4).tolist() * np.size(rot_list, axis=0)
    cellH= (lattice, position, numbers)
    dataset = get_symmetry_dataset(cellH,symprec=1e-5)
    # print(dataset)
    # print(cellG)
    # print(rot_list)
    # print(trans_list)
    Gnum = int(dataset['number'])
    return [Gnum, rot_list, tau_list]



def expand_magmoms(magmoms_str):
    expanded_parts = []
    parts = magmoms_str.split()
    
    for part in parts:
        if '*' in part:
            count, value = part.split('*')
            count = int(count)
            value = float(value)
            expanded_parts.extend([value] * count)
        else:
            expanded_parts.append(float(part))
    
    return expanded_parts

def read_magmom(file_path):
    magmom = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'MAGMOM' in line:
                # 去除关键字和等号，只保留数据部分
                data_str = line.split('=')[1].strip()
                # 处理类似于18*0的部分
                expanded_data_list = expand_magmoms(data_str)
                # print(expanded_data_list)
                # 组合成3x1的子数组，确保总数是3的倍数
                if len(expanded_data_list) % 3 != 0:
                    raise ValueError("MAGMOM values are not a multiple of 3")
                magmom = [expanded_data_list[i:i+3] for i in range(0, len(expanded_data_list), 3)]
                break
    return np.array(magmom)



def print_matlist(matlist):
    for i, mat in enumerate(matlist):
        print(i+1, ':')
        # print(mat)
        print_mat(np.array(mat))

def print_taulist(matlist):
    for i, mat in enumerate(matlist):
        print(i+1, ':')
        # print(mat)
        print_tau(np.array(mat))


def print_tau(mat, wfile=None, print1=True):
    dim = np.shape(mat)[0]
    strg = '['
    for row in range(dim):
        # strg += '[ '
        if 1:
            num = mat[row]
            if abs(num) < 1e-8:
                strg += '     0      '
            elif abs(num.real) < 1e-8:
                strg += ' %4.3f *I ' % (num.imag)
            elif abs(num.imag) < 1e-8:
                strg += ' %4.3f ' % (num.real)
            else:
                strg += ' (%4.3f + %4.3f *I) ' % (num.real, num.imag)
    strg += ' ]'
    if print1:
        print(strg, file=wfile)
    else:
        return strg

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Identify a spin space group from POSCAR or identifySSG.mcif')
    parser.add_argument('--symprec', type=float, default=1e-4, help='atomic-position tolerance in lattice length units')
    parser.add_argument('--mag-tol', type=float, default=1e-4, help='absolute moment tolerance in input moment units')
    parser.add_argument('--translation-tol', type=float, default=1e-4, help='fractional-translation matching tolerance')
    parser.add_argument('--rotation-tol', type=float, default=1e-3, help='spatial-matrix matching tolerance')
    parser.add_argument('--spin-tol', type=float, default=1e-2, help='spin character/determinant matching tolerance')
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        _positive_tolerance(value, name)
    warnings.filterwarnings('ignore')
    if os.path.exists('POSCAR'):
        
        script_dir = os.path.dirname(os.path.abspath(__file__))


        pkl_file_path = os.path.join(script_dir, 'ssg_data/identify.pkl')
        file = pkl_file_path
        with open(file, 'rb') as f:
            ssg_list = pickle.load(f)

        cell = read_poscar('POSCAR')

        print('Reading POSCAR as input ......')
    elif os.path.exists('identifySSG.mcif'):
        cell = mcif2cell('identifySSG.mcif')
        print('Reading mcif as input......')
    
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_file_path = os.path.join(script_dir, 'ssg_data/identify.pkl')
        file = pkl_file_path
        with open(file, 'rb') as f:
            ssg_list = pickle.load(f)
    else:
        # print('Have no input files !!!')
        raise ValueError('cant find input file!!! use POSCAR or identifySSG.mcif as input')
    
    lattice = cell[0].copy()
    position = cell[1].copy()
    numbers = cell[2].copy()
    mag = cell[3].copy()
    if findDimension(cell, args.mag_tol) == 0:
        print('It is a non-magnetic structure')
        cell_nonmag = (lattice, position, numbers)
        nonmag_sym = get_symmetry_dataset(cell_nonmag, symprec=args.symprec)
        print('Operations:')
        print('space rotations:')
        print_matlist(nonmag_sym['rotations'])
        print('==================')
        print('translations:')
        print_taulist(nonmag_sym['translations'])
        print('==================')
        # print(operations)
        print('The SG number:',nonmag_sym['number'],'(',nonmag_sym['international'],')')
    else:
        operations = findAllOp(cell, tol=args.symprec, mag_tol=args.mag_tol)
        print('Operations:')
        print('spin rotations:')
        print_matlist(operations['spin'])
        print('==================')
        print('space rotations:')
        print_matlist(operations['RotC'])
        print('==================')
        print('translations:')
        print_taulist(operations['TauC'])
        print('==================')
        # print(operations)
        ssgnum = search4ssg(cell, ssg_list, tol=args.symprec, operations=operations,
                            mag_tol=args.mag_tol, translation_tol=args.translation_tol,
                            rotation_tol=args.rotation_tol, spin_tol=args.spin_tol)
        print('The SSG number:')
        print(ssgnum)
        print('More information about this group can be found at')
        print('https://cmpdc.iphy.ac.cn/ssg/ssgs/'+ssgnum)




if __name__ == '__main__':
    main()
