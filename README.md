# CRGNN

Paper ： 
## Environments

Implementing environment: GeForce RTX™ 3090 24GB (GPU)

## Requirements

pytorch>=1.8.1

ogb=1.3.2

numpy=1.21.2

cogdl (latest version)

## Training

### GAMLP+RLU+SCR

For **ogbn-products**:

###### Params: 3335831

```bash
python pre_processing.py --num_hops 5 --dataset ogbn-products

python main.py --use-rlu --method R_GAMLP_RLU --stages 400 300 300 300 300 300 --train-num-epochs 0 0 0 0 0 0 --threshold 0.85 --input-drop 0.2 --att-drop 0.5 --label-drop 0 --pre-process --residual --dataset ogbn-products --num-runs 10 --eval 10 --act leaky_relu --batch_size 50000 --patience 300 --n-layers-1 4 --n-layers-2 4 --bns --gama 0.1 --consis --tem 0.5 --lam 0.1 --hidden 512 --ema
```

### GAMLP+MCR

For **ogbn-products**:

###### Params: 3335831

```bash
python pre_processing.py --num_hops 5 --dataset ogbn-products

python main.py --use-rlu --method R_GAMLP_RLU --stages 800 --train-num-epochs 0 --input-drop 0.2 --att-drop 0.5 --label-drop 0 --pre-process --residual --dataset ogbn-products --num-runs 10 --eval 10 --act leaky_relu --batch_size 100000 --patience 300 --n-layers-1 4 --n-layers-2 4 --bns --gama 0.1 --tem 0.5 --lam 0.5 --ema --mean_teacher --ema_decay 0.999 --lr 0.001 --adap --gap 10 --warm_up 150 --top 0.9 --down 0.8 --kl --kl_lam 0.2 --hidden 512

```

### GIANT-XRT+GAMLP+MCR

Please follow the instruction in [GIANT](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt) to get the GIANT-XRT node features. 

For **ogbn-products**:

###### Params: 2144151

```bash
python pre_processing.py --num_hops 5 --dataset ogbn-products --giant_path " "

python main.py --use-rlu --method R_GAMLP_RLU --stages 800 --train-num-epochs 0 --input-drop 0.2 --att-drop 0.5 --label-drop 0 --pre-process --residual --dataset ogbn-products --num-runs 10 --eval 10 --act leaky_relu --batch_size 100000 --patience 300 --n-layers-1 4 --n-layers-2 4 --bns --gama 0.1 --tem 0.5 --lam 0.5 --ema --mean_teacher --ema_decay 0.99 --lr 0.001 --adap --gap 10 --warm_up 150 --kl --kl_lam 0.2 --hidden 256 --down 0.7 --top 0.9 --giant

```

### SAGN+MCR

For **ogbn-products**:

###### Params: 2179678

```bash
python pre_processing.py --num_hops 3 --dataset ogbn-products

python main.py --method SAGN --stages 1000 --train-num-epochs 0 --input-drop 0.2 --att-drop 0.4 --pre-process --residual --dataset ogbn-products --num-runs 10 --eval 10 --batch_size 100000 --patience 300 --tem 0.5 --lam 0.5 --ema --mean_teacher --ema_decay 0.99 --lr 0.001 --adap --gap 20 --warm_up 150 --top 0.85 --down 0.75 --kl --kl_lam 0.01 --hidden 512 --zero-inits --dropout 0.5 --num-heads 1  --label-drop 0.5  --mlp-layer 2 --num_hops 3 --label_num_hops 14
```

### GIANT-XRT+SAGN+MCR

Please follow the instruction in [GIANT](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt) to get the GIANT-XRT node features. 

For **ogbn-products**:

###### Params: 1154654

```bash
python pre_processing.py --num_hops 3 --dataset ogbn-products --giant_path " "

python main.py --method SAGN --stages 1000 --train-num-epochs 0 --input-drop 0.2 --att-drop 0.4 --pre-process --residual --dataset ogbn-products --num-runs 10 --eval 10 --batch_size 50000 --patience 300 --tem 0.5 --lam 0.5 --ema --mean_teacher --ema_decay 0.99 --lr 0.001 --adap --gap 20 --warm_up 100 --top 0.85 --down 0.75 --kl --kl_lam 0.02 --hidden 256 --zero-inits --dropout 0.5 --num-heads 1  --label-drop 0.5  --mlp-layer 1 --num_hops 3 --label_num_hops 9 --giant

```

### Use Optuna to search for C&S hyperparameters

We searched hyperparameters using Optuna on **validation set**.
```bash
python post_processing.py --file_name --search
```

### GAMLP+RLU+SCR+C&S
```bash
python post_processing.py --file_name --correction_alpha 0.4780826957236622 --smoothing_alpha 0.40049734940262954
```

### GIANT-XRT+SAGN+MCR+C&S
```bash
python post_processing.py --file_name --correction_alpha 0.42299283241438157 --smoothing_alpha 0.4294212449832242
```

## Node Classification Results:

Performance on **ogbn-products**(10 runs):
| Methods   | Validation accuracy  | Test accuracy  |
|  ----  | ----  |  ---- |
| SAGN+MCR  | 0.9325±0.0004 | 0.8441±0.0005  |
| GAMLP+MCR  | 0.9319±0.0003 | 0.8462±0.0003  |
| GAMLP+RLU+SCR  | 0.9292±0.0005 |  0.8505±0.0009  |
| GAMLP+RLU+SCR+C&S  | 0.9304±0.0005 |  **0.8520±0.0008**  |
| GIANT-XRT+GAMLP+MCR  | 0.9402±0.0004 | 0.8591±0.0008 |
| GIANT-XRT+SAGN+MCR  | 0.9389±0.0002 |  0.8651±0.0009  |
| GIANT-XRT+SAGN+MCR+C&S  | 0.9387±0.0002 | **0.8673±0.0008** |


## Citation
Our paper:
```

```

GIANT paper:
```
@article{chien2021node,
  title={Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction},
  author={Eli Chien and Wei-Cheng Chang and Cho-Jui Hsieh and Hsiang-Fu Yu and Jiong Zhang and Olgica Milenkovic and Inderjit S Dhillon},
  journal={arXiv preprint arXiv:2111.00064},
  year={2021}
}
```
GAMLP paper:
```
@article{zhang2021graph,
  title={Graph attention multi-layer perceptron},
  author={Zhang, Wentao and Yin, Ziqi and Sheng, Zeang and Ouyang, Wen and Li, Xiaosen and Tao, Yangyu and Yang, Zhi and Cui, Bin},
  journal={arXiv preprint arXiv:2108.10097},
  year={2021}
}
```


SAGN paper:

```
@article{sun2021scalable,
  title={Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training},
  author={Sun, Chuxiong and Wu, Guoshi},
  journal={arXiv preprint arXiv:2104.09376},
  year={2021}
}
```

C&S paper:

```
@inproceedings{
huang2021combining,
title={Combining Label Propagation and Simple Models out-performs Graph Neural Networks},
author={Qian Huang and Horace He and Abhay Singh and Ser-Nam Lim and Austin Benson},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=8E1-f3VhX1o}
}
```
