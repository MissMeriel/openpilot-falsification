# Finding Property Violations through Network Falsification:  Challenges, Adaptations and Lessons Learned from OpenPilot

## Getting Started

ACM publication (DOI 10.1145/3551349.3559500): [https://dl.acm.org/doi/10.1145/3551349.3559500](https://dl.acm.org/doi/10.1145/3551349.3559500)

### What's in this repo?
1. [Extended results](Extended-results.md) from the paper
	1. 	Original output for all 10 images
	2. Baseline for all 10 images
	3. Counterexamples for all 10 images on all 3 safety properties
2. [How to reproduce](How-to-reproduce.md) results from the paper
    1. How to generate original images dataset
    2. How to generate baseline images
    3. How to generate counterexamples
    4. Description of image distance metrics
3. Links to other [large-scale networks of interest](Future-work.md) (candidate networks for further study):
	1. BigGAN
	2. AdmiralNet
	3. Detectron2
4. [Further documentation](documentation) of OpenPilot, including network input and output




## OpenPilot versioning

This project uses the most recent stable version of OpenPilot, determined by analyzing commits and building and testing several versions on a simple stretch of road. The most recent stable version is:

 
```buildoutcfg
OpenPilot (tag): 6be70a063dfc96b9e9f097f439bdc2e0be54d6d9
openpilot_docker (tag): sha256:35ff99c6004869d97d05a90fa77668f732596d74886f9a5c5fe5003233536b2f
Carla (Version): 0.9.12 (edited)
```

To read more on performing a full installation of OpenPilot, refer to [documentation](documentation) or [Installing OpenPilot](https://github.com/commaai/openpilot/tree/master/tools)

## Installing the extended DNNF

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
