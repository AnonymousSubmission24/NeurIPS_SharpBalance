# Sharpness-diversity tradeoff: improving flat ensembles with SharpBalance
Official PyTorch implementation of SharpBalance

## Introduction
__SharpBalance__ is an ensembling method that balances the ensemble **diversity** and **sharpness** of individual ensemble members. In this work, We discover a trade-off between sharpness and diversity: minimizing the sharpness in the loss landscape tends to diminish the diversity of individual members within the ensemble, adversely affecting the ensemble's improvement. The trade-off is justified through our theoretical analysis and experiments.
<p align="center">
  <img src="./assert/simulated_estimated.png" width="300" />
  <img src="./assert/teaser_cifar10_trade_off.png" width="221" /> 
</p>
SharpBalance aims to achieve the optimal balance by applying SAM to a carefully selected subset of the data, while performing standard optimization on the remaining samples. The pipeline of the algorithm is shown below
<p align="center">
  <img src="./assert/sharpbalance_diagram.png" width="500" />
</p>
Experiments show that SharpBalance improves the overall ensembling performance and mitigates the reduced ensembling improvement caused by sharpness-diversity trade-off.
<p align="center">
  <img src="./assert/benchmark_results.png" width="600" />
</p>

---
## Setup


## Minimal Example
```

```


--- 
### Script example of pruning llama2-7b using Magnitude-based pruning with our layer-wise pruning ratio
```

```

### Script example of pruning llama2-7b using Wanda with our layer-wise pruning ratio
```

```

### Script example of pruning llama2-7b using SparseGPT with our layer-wise pruning ratio
```

```


## Usage
We provide a quick overview of the arguments:  
```

```


**More details coming soon!**
