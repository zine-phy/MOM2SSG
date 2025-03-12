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
