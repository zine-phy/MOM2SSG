# identifySSG

This Python program can be used to identify the Spin-Space Group (SSG) of a magnetic structure.

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/zine-phy/MOM2SSG.git
    ```

2. **Extract the data in `ssg_data`:**

    ```bash
    tar -xzvf identify.pkl.tar.gz
    ```

3. **Install the necessary libraries:**

    Generally, installing `pymatgen` will install all the dependencies.

    ```bash
    pip install pymatgen==2022.0.17
    ```

4. **Install the package:**

    ```bash
    python setup.py install
    ```

## Usage

You can use `identifySSG` to identify the SSG of a magnetic structure. There are two ways to input the magnetic structure:

1. **Using `POSCAR` files:**  

    - The `POSCAR` file should follow the same format as the VASP input file, with an extension for magnetic moments.  
    - For magnetic atoms, each line in the atomic position section should include coordinates \((x_1, x_2, x_3)\) in direct (fractional) format, followed by the Cartesian magnetic moment components \((m_1, m_2, m_3)\).  
    - For non-magnetic atoms, you only need to provide the atomic coordinates without specifying the magnetic moment.  

    Example: Check the `example/` directory for a sample `POSCAR` file of \ch{Mn3Sn}. 
2. **Using an `mcif` file:**

    - Name the file `identifySSG.mcif`.

In the `example/` directory, we provide both a `POSCAR` file and an `mcif` file for \ch{Mn3Sn}. The `mcif` file is directly downloaded from [MAGNDATA](https://www.cryst.ehu.es/magndata/index.php?index=0.199). Users can run `identifySSG` in either the `mcif/` or `POSCAR/` folder, and both methods will yield the same SSG result: **194.1.6.1.P**.



## Notes on Space Group Settings

In the enumeration of SSGs, space groups are defined based on their ITA (International Tables for Crystallography) numbers. For instance, space group No. 62 is defined in its standard ITA setting as **Pnma**. Alternative settings like **Pmnb** are treated equivalently. 



## References

This program relies on the following libraries and data sources:

1. **Spglib:**  
   Used for symmetry analysis.  
   > Togo, A., Shinohara, K., & Tanaka, I. (2024). *Spglib: a software library for crystal symmetry search*. *Sci. Technol. Adv. Mater., Meth.*, 4(1), 2384822–2384836. DOI: [10.1080/27660400.2024.2384822](https://doi.org/10.1080/27660400.2024.2384822).

2. **Pymatgen:**  
   Used for materials analysis and point group operations.  
   > Brown, P. J., Nunez, V., Tasset, F., Forsyth, J. B., & Radhakrishna, P. (1990). *Determination of the magnetic structure of Mn3Sn using generalized neutron polarization analysis*. *Computational Materials Science*, 2(47), 9409. DOI: [10.1088/0953-8984/2/47/015](https://dx.doi.org/10.1088/0953-8984/2/47/015).

3. **SSG Data Source:**  
   The spin-space group (SSG) data used in this program is based on the following work:  
   > Jiang, Y., Song, Z., Zhu, T., Fang, Z., Weng, H., Liu, Z.-X., Yang, J., & Fang, C. (2024). *Enumeration of Spin-Space Groups: Toward a Complete Description of Symmetries of Magnetic Orders*. *Phys. Rev. X, 14*(3), 031039. DOI: [10.1103/PhysRevX.14.031039](https://link.aps.org/doi/10.1103/PhysRevX.14.031039).

## License

This project is licensed under the terms of the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software in compliance with the license terms.

## Matching diagnostics

`findAllOp` determines the spin-space operations. `search4ssg` then filters database candidates by `G`, `H`, the lattice and point-group indices, and the spin point-group label. Every candidate is checked against its real-space and spin operations, including a unique candidate.

For debugging a POSCAR, the Python API can reuse the operations already found and return details:

```python
import pickle
from pathlib import Path
from identifySSG.identifySSG import read_poscar, findAllOp, search4ssg

cell = read_poscar('POSCAR')
operations = findAllOp(cell, 1e-4)
with (Path(__import__('identifySSG').__file__).parent / 'ssg_data/identify.pkl').open('rb') as stream:
    database = pickle.load(stream)
details = search4ssg(cell, database, operations=operations, return_details=True)
print(details)
```

The `status` field reports `matched`, `ambiguous`, `no_candidates`, or `no_equivalent_operations`. `stage` identifies the first successful setting search: `spglib`, `automorphism`, `translation_coset`, `extra_origin_shift`, or `triclinic_index_two`. The default call still returns the SSG number string or `need more loop`.

Translations are compared modulo integer lattice vectors without decimal rounding. The matcher composes the spglib setting with space-group automorphisms, checks origin shifts in eighth-cell steps when needed, and constructs an integral basis transformation directly for index-two sublattices of space group No. 2. Regression cases are in `tests/test_matching_regression.py`.

## Numerical tolerances

The magnetic-moment tolerance is separate from the spatial tolerance. The CLI
accepts these options (all must be finite and positive):

| Option | Python API | Default | Meaning |
| --- | --- | --- | --- |
| `--symprec` | `tol` | `1e-4` | Atomic-position tolerance for the initial spatial symmetry search; lattice length units |
| `--mag-tol` | `mag_tol` | `1e-4` | Absolute moment-vector distance; input moment units (usually Bohr magnetons) |
| `--translation-tol` | `translation_tol` | `1e-4` | Fractional-translation distance modulo integers, in input-cell coordinates |
| `--rotation-tol` | `rotation_tol` | `1e-3` | Frobenius distance of spatial rotation matrices during database matching |
| `--spin-tol` | `spin_tol` | `1e-2` | Difference of spin traces, or determinants for collinear structures, during database matching |

```bash
python -m identifySSG.identifySSG --symprec 1e-4 --mag-tol 1e-4
```

```python
operations = findAllOp(cell, tol=1e-4, mag_tol=1e-5)
details = search4ssg(cell, database, tol=1e-4, mag_tol=1e-5,
                    operations=operations, return_details=True)
```

Use the same tolerances for operation discovery and number identification.
Operations produced by this version record `mag_tol`, `symprec`, and `dimension`;
reusing them with conflicting settings raises an error. Legacy operation
dictionaries without this metadata remain accepted; callers must ensure that
they were computed consistently.

Magnetic dimension uses SVD subspaces through the origin. A structure is
collinear/coplanar only when **every atom's** distance to the best line/plane is
below `mag_tol / 2`. This bounds the change under reflection by `mag_tol` and
avoids dependence on the number of repeated atoms. All moments below `mag_tol`
are treated as nonmagnetic. The CLI returns an ordinary space group for that
case; the magnetic-operation API raises a clear error.

Moment values retain their input precision during atom sorting. Collinear
search uses the two representatives `I` and `-I`, modulo the spin-only
stabilizer; planar lifting uses an SVD plane normal. Input moments are not
projected or overwritten. Details include singular values, maximum residuals
for ranks 0/1/2, and the tolerances used.

These options do not replace every internal numerical constant. Existing
coordinate sorting uses a `1e-5` fractional grid, and synthetic-cell space-group
classification and point-group classification retain their internal thresholds.
`--symprec` therefore does not claim to control all internal geometric checks.
Very small `mag_tol` can resolve genuine canting or input noise and change the
magnetic dimension; check a tolerance range when the input precision is uncertain.

Run `tests/test_matching_regression.py` for operation-equivalence regressions.
`tests/test_tolerances.py` additionally tests magnetic precision and the four
real failures; it requires `MOM2SSG_PROJECT` to point to the project containing
`need_more_loop/` and `MOM2SSG_RUN` to a writable result directory.
