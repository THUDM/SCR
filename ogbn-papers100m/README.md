# SCR for ogbn-papers100M

## Environments

Implementing environment: GeForce RTX™ 3090 24GB (GPU)

## Training

### GAMLP+SCR

For **ogbn-papers100M**:

```bash
python main_papers.py --method JK_GAMLP --stages 1000 --train-num-epochs 0 --dataset ogbn-papers100M --eval-every 1 --act sigmoid --batch 5000 --eval-batch 50000 --patience 60 --n-layers-1 4 --n-layers-2 6 --num-hops 12 --input-drop 0 --att-drop 0.5 --pre-process --hidden 1280 --lr 0.001 --use-label --label-num-hops 9 --label-drop 0.3 --scr --adap --kl --top 0.9 --down 0.8 --warm_up 100 --gap 10 --kl_lam 0.03 --ema_decay 0.0
```

### GAMLP+RLU+SCR

For **ogbn-papers100M**:

```bash
python main_papers.py --method JK_GAMLP --stages 400 400 400 400 --train-num-epochs 0 0 0 0 --threshold 0.9 --dataset ogbn-papers100M --eval-every 1 --act sigmoid --batch 5000 --eval-batch 50000 --patience 60 --n-layers-1 4 --n-layers-2 6 --num-hops 12 --input-drop 0 --att-drop 0.5 --pre-process --hidden 1280 --lr 0.001 --use-label --label-num-hops 9 --label-drop 0.3 --temp 0.001 --consis --lam 0.1 
```

### GAMLP+SCR-m

For **ogbn-papers100M**:

```bash
python main_papers.py --method JK_GAMLP --stages 1000 --train-num-epochs 0 --dataset ogbn-papers100M --eval-every 1 --act sigmoid --batch 5000 --eval-batch 50000 --patience 60 --n-layers-1 4 --n-layers-2 6 --num-hops 12 --input-drop 0 --att-drop 0.5 --pre-process --hidden 1280 --lr 0.001 --use-label --label-num-hops 9 --label-drop 0.3 --scr --adap --kl --top 0.9 --down 0.8 --warm_up 100 --gap 10 --kl_lam 0.03 --ema_decay 0.99

```


## Node Classification Results:

Performance on **ogbn-papers100M**(3 runs):
| Methods   | Validation accuracy  | Test accuracy  |
|  ----  | ----  |  ---- |
| GAMLP+SCR  | 0.7190±0.0007 | 0.6814±0.0008  |
| GAMLP+SCR-m  | 0.7186±0.0008 | 0.6816±0.0012  |
| GAMLP+RLU+SCR  | 0.7188±0.0007 |  **0.6842±0.0015**  |

