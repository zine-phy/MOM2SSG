"""Operation-level regression cases for SSG setting equivalence.

Run with the project's Python environment and PYTHONPATH pointing to the
working source tree. The database is read from the installed package tree.
"""

import copy
import pickle
from pathlib import Path

import numpy as np

from identifySSG import identifySSG as module
from identifySSG.SG_utils import identify_SG_lattice


DATA = Path(module.__file__).parent / "ssg_data/identify.pkl"
with DATA.open("rb") as stream:
    database = pickle.load(stream)


def inverse_affine(rots, taus, A, a):
    inverse = np.linalg.inv(A)
    return ([inverse @ R @ A for R in rots],
            [inverse @ (R @ a + t - a) for R, t in zip(rots, taus)])


def synthetic_operations(record, A, a):
    gid, ik, it = (int(v) for v in record["search"].split("."))
    q_rots, q_taus = inverse_affine(record["QRotC"], record["QTauC"], A, a)
    prim = identify_SG_lattice(gid)[1]
    supercell = np.asarray(record["superCell"])
    h_rots = list(record["HRotC"]) + [np.eye(3)] * 3
    h_taus = list(record["HTauC"]) + [prim @ supercell[:, i] for i in range(3)]
    h_rots, h_taus = inverse_affine(h_rots, h_taus, A, a)
    return {
        "Gnum": gid, "Hnum": record["Hid"], "Ik": ik, "It": it,
        "QLabel": record["eqvPg"], "RotC": q_rots, "TauC": q_taus,
        "spin": record["URot"][0], "HRotC": h_rots, "HTauC": h_taus,
        "transformation_matrix": np.eye(3), "original_shift": np.zeros(3),
    }


def run_case(number, A, a, expected_stage, reported_transform=None):
    target = next(x for x in database if x["ssgNum"] == number)
    peers = [x for x in database if all(x[key] == target[key] for key in
             ("dim", "search", "Hid", "eqvPg"))]
    before = [(len(x["HRotC"]), len(x["HTauC"])) for x in peers]
    operations = synthetic_operations(target, np.asarray(A), np.asarray(a))
    if reported_transform is not None:
        operations["transformation_matrix"] = np.asarray(reported_transform)
    dim = target["dim"]
    cell = (np.eye(3), np.zeros((dim, 3)), [1] * dim, np.eye(3)[:dim])
    details = module.search4ssg(cell, peers, operations=operations, return_details=True)
    after = [(len(x["HRotC"]), len(x["HTauC"])) for x in peers]
    assert before == after, "matching modified database records"
    assert details["status"] == "matched", details
    assert details["number"] == number, details
    assert details["stage"] == expected_stage, details
    assert module.search4ssg(cell, copy.deepcopy(peers), operations=operations) == number
    print(number, details["stage"], details["transforms_tested"])


assert module.same_translation_mod_lattice([0.125, 0.875, 1.0], [1.125, -0.125, 0.0])
assert not module.same_translation_mod_lattice([0.125, 0, 0], [0.13, 0, 0])

run_case(
    "15.1.4.5",
    [[-1, 0, -1], [0, 1, 0], [0, 0, -1]],
    [0.25, 0.25, 0.5],
    "automorphism",
)
run_case("12.1.4.1", np.eye(3), [0.25, 0, 0], "extra_origin_shift")
run_case(
    "65.2.1.1.L",
    [[-2, 2, 0], [0, 0, -1], [-2, -2, 0]],
    [0, 0, 0],
    "spglib",
    reported_transform=[[-2, 2, 0], [0, 0, -1], [-2, -2, 0]],
)
crreo4 = next(x for x in database if x["ssgNum"] == "12.2.2.9.P")
crreo4_basis = identify_SG_lattice(12)[1] @ np.asarray(crreo4["superCell"])
run_case("12.2.2.9.P", crreo4_basis, [0, 0, 0.5],
         "extra_origin_shift", reported_transform=crreo4_basis)
run_case("201.1.1.1.L", np.eye(3), [0.375, 0.375, 0.375], "extra_origin_shift")
triclinic_operations = {
    "Gnum": 2,
    "Ik": 2,
    "transformation_matrix": np.diag([2, -2, -1]),
    "HRotC": [np.eye(3), np.eye(3)],
    "HTauC": [np.zeros(3), np.array([0.5, 0.5, 0])],
}
triclinic_candidate = {"superCell": np.diag([1, 1, 2])}
triclinic_transform = module._triclinic_index_two_setting(
    triclinic_candidate, triclinic_operations)
assert round(abs(np.linalg.det(triclinic_transform))) == 1
assert np.array_equal((np.array([0, 0, 1]) @ triclinic_transform) % 2,
                      np.array([1, 1, 0]))
print("matching regression checks passed")
