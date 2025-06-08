# Genetic Programming for Symbolic Regression

*A complete starter repo for COIT29224 Assessment 3.*

```bash
# 1. create env & install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. supply dataset (optional)
#   → drop Table_1.csv into gp/data.csv (three columns: x,y,target)

# 3. run experiment
python main.py  # uses defaults (pop=100, gens=50, etc.)

# 4. tweak hyper‑params
python - <<'PY'
from gp import run_gp
best, history = run_gp(pop_size=200, generations=100, mse_threshold=0.05)
print(best)
PY
```

## Visualisations
- **Training curve:** automatically pops up after run (matplotlib line chart).
- **3‑D surfaces:** GP‑predicted vs. true dataset (requires ≥9 points).
  Useful for screenshots in the report.