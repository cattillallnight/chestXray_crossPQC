# Paper tables (generated; do not edit by hand)

Mean±std over seeds {42,43,44}. Discrimination is secondary. Temperature scaling is fit on CheXpert validation logits and applied as logits/T. A run is T-degenerate if T hit the optimizer clamp (0.001 or 100).

## Primary tasks

| Task | Head | AUROC CheXpert | ECE raw→cal CheXpert | AUROC VinDr-bal | ECE raw→cal VinDr-bal | n T-deg |
|---|---|---|---|---|---|---|
| No finding | MLP | 0.837±0.003 | 0.320±0.069→0.317±0.105 | 0.736±0.008 | 0.250±0.081→0.207±0.038 | 0 |
| No finding | Product PQC | 0.820±0.007 | 0.239±0.031→0.153±0.087 | 0.669±0.003 | 0.124±0.011→0.203±0.049 | 0 |
| No finding | Entangling PQC | 0.813±0.010 | 0.379±0.038→0.375±0.050 | 0.725±0.015 | 0.200±0.021→0.151±0.077 | 0 |
| Cardiomegaly | MLP | 0.681±0.005 | 0.276±0.082→0.219±0.134 | 0.774±0.012 | 0.187±0.069→0.263±0.131 | 0 |
| Cardiomegaly | Product PQC | 0.637±0.036 | 0.315±0.076→0.262±0.160 | 0.772±0.028 | 0.105±0.091→0.152±0.168 | 0 |
| Cardiomegaly | Entangling PQC | 0.679±0.006 | 0.422±0.048→0.376±0.001 | 0.765±0.021 | 0.093±0.021→0.039±0.064 | 1 |
| Pleural effusion | MLP | 0.761±0.004 | 0.112±0.036→0.092±0.034 | 0.665±0.018 | 0.160±0.076→0.118±0.077 | 0 |
| Pleural effusion | Product PQC | 0.752±0.018 | 0.072±0.005→0.075±0.012 | 0.643±0.010 | 0.178±0.014→0.187±0.024 | 0 |
| Pleural effusion | Entangling PQC | 0.748±0.011 | 0.105±0.032→0.100±0.036 | 0.619±0.035 | 0.105±0.051→0.091±0.030 | 0 |
| Consolidation | MLP | 0.652±0.030 | 0.417±0.047→0.386±0.047 | 0.682±0.035 | 0.107±0.040→0.115±0.034 | 1 |
| Consolidation | Product PQC | 0.642±0.026 | 0.362±0.102→0.300±0.196 | 0.714±0.012 | 0.085±0.029→0.200±0.134 | 2 |
| Consolidation | Entangling PQC | 0.613±0.033 | 0.407±0.011→0.408±0.021 | 0.604±0.056 | 0.073±0.029→0.096±0.029 | 0 |

## Appendix: pneumonia_proxy (ontology shift; not a pneumonia detector)

| Head | AUROC CheXpert | ECE raw→cal CheXpert | AUROC VinDr-bal | ECE raw→cal VinDr-bal | n T-deg |
|---|---|---|---|---|---|
| MLP | 0.662±0.071 | 0.467±0.048→0.436±0.041 | 0.629±0.044 | 0.059±0.050→0.024±0.021 | 1 |
| Product PQC | 0.625±0.072 | 0.338±0.053→0.099±0.093 | 0.582±0.062 | 0.113±0.073→0.446±0.029 | 2 |
| Entangling PQC | 0.711±0.032 | 0.438±0.038→0.437±0.048 | 0.609±0.025 | 0.087±0.033→0.083±0.049 | 0 |
