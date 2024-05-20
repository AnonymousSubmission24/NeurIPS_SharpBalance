# Sharpness-diversity tradeoff: improving flat ensembles with SharpBalance
Official PyTorch implementation of SharpBalance

## Introduction
SharpBalance is an ensembling method that balances the ensemble diversity and sharpness of individual ensemble members. In this work, We discover a trade-off between sharpness and diversity: minimizing the sharpness in the loss landscape tends to diminish the diversity of individual members within the ensemble, adversely affecting the ensemble's improvement. The trade-off is justified through our theoretical analysis and verified empirically through extensive experiments.

SharpBalance aims to achieve the optimal balance by applying SAM to a carefully selected subset of the data, while performing standard optimization on the remaining samples.

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
