# Genetic Programming for Symbolic Regression

*A complete starter repo for COIT29224 Assessment 3.*

## 1. create env & install deps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

```

## To run the project: 
```
python main.py --pop_size 500 --generations 300 \
               --max_depth 8 --mutation_rate 0.2
```


## Visualisations
- **Training curve:** automatically pops up after run (matplotlib line chart).
- **3‑D surfaces:** GP‑predicted vs. true dataset (requires ≥9 points).
  Useful for screenshots in the report.