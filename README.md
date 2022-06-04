# Spectral Adversarial Training for Robust Graph Neural Network

# Requirements
- torch == 1.9.0
- graphgallery

Install graphgallery:
```bash
git clone https://github.com/EdisonLeeeee/GraphGallery.git && cd GraphGallery
pip install -e . --verbose
```

# Reproduction
```bash
python main.py --dataset cora --backbone SSGC --lr 0.2 --k 30 --K 5
python main.py --dataset cora_ml --backbone SSGC --lr 0.2 --k 30 --K 5
python main.py --dataset citeseer --backbone SSGC --lr 0.2 --k 30 --K 5
python main.py --dataset pubmed --backbone SSGC --lr 0.2 --k 150 --K 5

python main.py --dataset cora --backbone SGC --lr 0.01 --k 30 --K 2
python main.py --dataset cora_ml --backbone SGC --lr 0.01 --k 30 --K 2
python main.py --dataset citeseer --backbone SGC --lr 0.01 --k 30 --K 2
python main.py --dataset pubmed --backbone SGC --lr 0.01 --k 150 --K 2


python main.py --dataset cora --backbone GCN --lr 0.01 --k 30
python main.py --dataset cora_ml --backbone GCN --lr 0.01 --k 30
python main.py --dataset citeseer --backbone GCN --lr 0.01 --k 30
python main.py --dataset pubmed --backbone GCN --lr 0.01 --k 150

```