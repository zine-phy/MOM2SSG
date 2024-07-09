# identifySSG

This Python program can be used to identify the Spin-Space Group (SSG) of a magnetic structure.

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone [<repository_url>](https://github.com/zine-phy/MOM2SSG.git)
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

1. **Using `POSCAR` and `INCAR.mag` files:**

    - `POSCAR` should follow the same format as the VASP input file.
    - `INCAR.mag` should provide the magnetic moments of the magnetic material.

2. **Using an `mcif` file:**

    - Name the file `identifySSG.mcif`.
