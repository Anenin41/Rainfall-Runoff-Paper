# Introduction
The script `symbo.py` is an upgrade of the old `integral_solver.py` script (not listed here), both of which aim to compute the vectors, matrices and tensors that exist in the new Shallow Water Moment model with Rainfall and Infiltration terms.

# How it Works
The crude implementation of the old script, even though correct, is very slow. This is because it used the `sympy` functions `integrate()` and `simplify()`. In practice, even though these functions work properly for the script and its calculations, they are very powerful, but also very costly, to be used only for the shifted Legendre polynomials. 

To solve this problem, a new symbolic representation of these formulas was found and implemented from the package's documentation. Running speed comparisons between the old and the new script showcases that this modified representation is approximately 30 times faster than what was previously implemented.

This script was encoded with CLI printability in mind. This means that the order of the model `N` should remain small, so the `stdout` can printed neatly into the terminal. In case this needs to be used as part of a solver suite, **then it needs refactoring**.

## Convention
Because the Shallow Water Moment models use the shifted and scaled Legendre polynomials as basis functions, in order to use the `legendre()` command of the package, the following convention takes place:

```
phi_n(z) = P_n(1 - 2 z), z in [0, 1]
```

## Output
- Vectors: `r`, `s`
- Matrices: `E`, `F`, `C`
- Tensors: `A`, `B`

## Key acceleration ideas:
- All integrands are polynomials in z. As such, it is sound to integrate exactly via the polynomial anti-derivative instead of SymPy's general integrate/simplify scheme.
- The A-tensor is computed via Wigner's 3j symbol, a closed-form identity, which connects the Legendre basis with a mathematical structure from spherical harmonics [[1]](https://mathoverflow.net/questions/450619/generalized-wigner-3-j-symbol-and-legendre-functions), [[2]](https://www.theoretical-physics.com/dev/math/spherical-harmonics.html).

# How to Use
Before running the script, it is necessary to install its dependencies. This can happen by following the commands listed on the block below. 

```bash
python3 -m venv .venv/      # Linux 
source .venv/bin/activate   # Activate the virtual environment (Linux)
pip install -r requirements.txt
```

The script uses the `argparse` library to fetch arguments exactly from the command line and print the results the user precisely needs, without flooding the CLI. Some examples include

```bash
python3 symbo.py [--h]      # Help message
python3 symbo.py --A --N 5
python3 symbo.py --B --N 5
python3 symbo.py --r --s --E --F --C --N 5
python3 symbo.py --all --N 5
```
