# openpilot-falsification
Finding Property Violations through Network Falsification:  Challenges, Adaptations and Lessons Learned from OpenPilot

# Getting Started

## Installing DNNF

```bash
git clone git@github.com:MissMeriel/dnnv.git
cd DNNV
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip flit
flit install -s
git checkout -b openpilot
pip install -e .
cd ..
git clone git@github.com:MissMeriel/dnnf.git
git checkout -b openpilot
pip install -e .
```

## Submitting pull requests to DNNF

```bash
cd dnnf
pytest
black <path/to/file>
git add <path/to/file>
git commit -S -m 'enhancement: Descriptive message'
```