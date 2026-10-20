"""Tolerance boundaries, invariance, and end-to-end real-case checks."""
import json
import os
import pickle
from pathlib import Path
import numpy as np
from identifySSG import identifySSG as m

def cell(moments):
    moments = np.asarray(moments, dtype=float)
    return (np.eye(3), np.zeros_like(moments), [1] * len(moments), moments)

for moments, expected in [([[1, 1e-7, 0], [-1, 1e-7, 0]], 1),
                          ([[1, 1e-3, 0], [-1, 1e-3, 0]], 2),
                          (np.eye(3), 3), (np.zeros((3, 3)), 0)]:
    c = cell(moments)
    before = c[3].copy()
    assert m.findDimension(c) == expected
    assert np.array_equal(c[3], before)
    # Replication and orthogonal spin-coordinate changes preserve dimension.
    rotation, _ = np.linalg.qr(np.array([[1., 2, 3], [-2, 4, 1], [3, 1, -1]]))
    assert m.findDimension(cell(c[3] @ rotation)) == expected
    assert m.findDimension(cell(np.tile(c[3], (7, 1)))) == expected

assert m.findDimension(cell([[1, 1e-7, 0], [-1, 1e-7, 0]]), 1e-8) == 2
assert len(m.findQ(cell([[1, 0, 0], [-1, 0, 0]]))) == 2
assert m.identity_cell(cell([[1, 0, 0]]), cell([[1, 5e-5, 0]]), 1e-4)
assert not m.identity_cell(cell([[1, 0, 0]]), cell([[1, 5e-5, 0]]), 1e-6)
assert not m.identity_cell(cell([[1, 0, 0]]), cell([[1, 1e-7, 0]]), 1e-8)
assert np.array_equal(m.std_cell(cell([[1, 1e-9, 0]]))[3], [[1, 1e-9, 0]])
for invalid in (0, -1, float('nan'), float('inf')):
    try:
        m.findDimension(cell([[1, 0, 0]]), invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(f'accepted invalid tolerance {invalid}')
try:
    m.search4ssg(cell([[1, 0, 0]]), [], operations={'mag_tol': 1e-6})
except ValueError as e:
    assert 'different mag_tol' in str(e)
else:
    raise AssertionError('accepted stale operations')

project = Path(os.environ['MOM2SSG_PROJECT'])
with (Path(m.__file__).parent/'ssg_data/identify.pkl').open('rb') as f:
    database = pickle.load(f)
cases = {
    '0.1008_Sr2ErRuO6.POSCAR': '14.1.2.3.L',
    '1.459_CeFe3(BO3)4.POSCAR': '155.2.1.1.L',
    '1.485_Mn3TeO6.POSCAR': '14.2.1.1.L',
    '1.808_BaFeO2.96.POSCAR': '187.2.1.1.L',
}
results = []
for name, expected in cases.items():
    c = m.read_poscar(str(project/'need_more_loop'/name))
    # A range around the default must give the same verified number.
    for tolerance in (1e-6, 1e-5, 1e-4):
        operations = m.findAllOp(c, 1e-4, mag_tol=tolerance)
        result = m.search4ssg(c, database, operations=operations,
                             mag_tol=tolerance, return_details=True)
        results.append(dict(file=name, **result))
        print(name, tolerance, result['status'], result['number'], flush=True)
        assert result['status'] == 'matched' and result['number'] == expected, result
        assert result['dimension_details']['dimension'] == 1
        assert result['h_matches'] and result['q_matches']
Path(os.environ['MOM2SSG_RUN'], 'tolerance_sweep.json').write_text(json.dumps(results, indent=2))
print('tolerance tests passed', flush=True)
