# SCR for ogbn-mag

## Environments

Implementing environment: GeForce RTX™ 3090 24GB (GPU)

## Training

### NARS_GAMLP+SCR


```bash
python main.py --use-rlu --method JK_GAMLP_RLU --stages 1000 --train-num-epochs 0 --input-drop 0.1 --att-drop 0 --label-drop 0 --pre-process --residual --dataset ogbn-mag --num-runs 10 --eval 10 --act leaky_relu --batch 10000 --patience 100 --n-layers-1 4 --n-layers-2 4 --label-num-hops 3 --bns --gama 10 --scr --ema_decay 0.0 --adap --kl --top 0.5 --down 0.4 --warm_up 60 --gap 10 --kl_lam 0.025
```

### NARS_GAMLP+RLU+SCR

```bash
python main.py --use-rlu --method JK_GAMLP_RLU --stages 250 200 200 200 200 200 200 --train-num-epochs 0 0 0 0 0 0 0 --threshold 0.4 --input-drop 0.1 --att-drop 0 --label-drop 0 --pre-process --residual --dataset ogbn-mag --num-runs 10 --eval 10 --act leaky_relu --batch 10000 --patience 300 --n-layers-1 4 --n-layers-2 4 --label-num-hops 3 --bns --gama 10 --con --lam 0.07
```

### NARS_GAMLP+SCR-m

```bash
python main.py --use-rlu --method JK_GAMLP_RLU --stages 1000 --train-num-epochs 0 --input-drop 0.1 --att-drop 0 --label-drop 0 --pre-process --residual --dataset ogbn-mag --num-runs 10 --eval 10 --act leaky_relu --batch 10000 --patience 100 --n-layers-1 4 --n-layers-2 4 --label-num-hops 3 --bns --gama 10 --scr --ema_decay 0.99 --adap --kl --top 0.5 --down 0.4 --warm_up 60 --gap 10 --kl_lam 0.025

```


## Node Classification Results:

Performance on **ogbn-mag**(10 runs):
| Methods   | Validation accuracy  | Test accuracy  |
|  ----  | ----  |  ---- |
| NARS_GAMLP+SCR  | 0.5654±0.0021 | 0.5432±0.0018  |
| NARS_GAMLP+SCR-m  | 0.5590±0.0028 | 0.5451±0.0019  |
| NARS_GAMLP+RLU+SCR  | 0.5734±0.0035 |  **0.5631±0.0021**  |
