# Sparse-Random-Networks-for-Communication-Efficient-Federated-Learning
PyTorch implementation of the FedPM framework by the authors of the ICLR 2023 paper "Sparse Random Networks for Communication-Efficient Federated Learning".

> [Sparse Random Networks for Communication-Efficient Federated Learning](https://arxiv.org/pdf/2209.15328.pdf) <br/>
>[Berivan Isik](https://sites.google.com/view/berivanisik), [Francesco Pase](https://sites.google.com/view/pasefrance), [Deniz Gunduz](https://www.imperial.ac.uk/people/d.gunduz), [Tsachy Weissman](https://web.stanford.edu/~tsachy/), [Michele Zorzi](https://signet.dei.unipd.it/zorzi/) <br/>
> International Conference on Learning Representations (ICLR), 2023. <br/>


## Environment setup:
Packages can be found in `fedpm.yml`.

## Training:
Set the `params_path` in `main.py` to the the path of the `{}.yaml` file with the desired model and dataset. The default parameters can be found in the provided `{}.yaml` files. To train the model, run:

```
python3 main.py
```

## References
If you find this work useful in your research, please consider citing our paper:
```
@inproceedings{
isik2023sparse,
title={Sparse Random Networks for Communication-Efficient Federated Learning},
author={Berivan Isik and Francesco Pase and Deniz Gunduz and Tsachy Weissman and Zorzi Michele},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=k1FHgri5y3-}
}
```
