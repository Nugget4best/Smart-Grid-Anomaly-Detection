# Datasets

## 1. UCI Electrical Grid Stability Simulated Dataset

**Source:** UCI Machine Learning Repository

A simulated dataset of a 4-node star electrical grid topology (1 producer, 3 consumers).

- **Samples:** 10,000
- **Features:** 12 predictive + 2 targets (continuous `stab` + categorical `stabf`)
- **Task:** Binary classification — stable vs. unstable grid state

### Download
1. Visit: https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data
2. Download `Data_for_UCI_named.csv`
3. Place it in this `data/` directory

### Citation
Arzamasov, V., Bohm, K., & Jochem, P. (2018).
"Towards Concise Models of Grid Stability."
IEEE PES Innovative Smart Grid Technologies Conference Europe (ISGT-Europe).

## 2. Synthetic Power Plant Operational Data

Generated programmatically by `src/data_loader.py` — no download needed.

Simulates a gas-turbine power plant with 13 operational parameters and 5 fault modes:
- Overload, Voltage sag, Frequency deviation, Thermal runaway, Sensor drift

The generation code is deterministic (seeded) for full reproducibility.

## Note
Raw data files are excluded from this repository via `.gitignore`.
