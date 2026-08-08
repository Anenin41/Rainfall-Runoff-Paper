# Overview
This folder contains the rainfall-infiltration extension of the existing one-dimensional Shallow Water Moment Equations (SWME) solver. The purpose of this extension is to add recharge-related physics to the pre-existing suite in a modular way, without rewriting the original numerical core of the repository. 

The main idea is to keep the transport and discretization machinery of the existing solver instact, and to introduce rainfall, infiltration and related mixing-friction source terms through a separate grouped submodule. This makes it easier to develop, test, and merge this source code with the baseline suite. 

In practice, this extension is meant to support models in which the base SWME is augmented by:
- a mass production source term of the form `R - I`,
- additional momentum contributions induced by rainfall and infiltration,
- and, if needed, extra source terms in the moment evolution equation.

The first implementation is intentionally simple. Everything is designed to be compatible with the current solver structure and to provide a clean point of entry for further and more advanced development.

# Prerequisites
In order for the baseline solver, as well as this extension to work, it is necessary to download and install the package prerequisites. This can be done in an easy manner by following the command block below:

```bash
python3 -m venv .venv               # Create an environment (Linux)
source .venv/bin/activate           # Activate it (Linux)
pip install -r requirements.txt     # Install runtime dependencies
```
**Tip:** Create the virtual environment at the root of the repository for ease of access.