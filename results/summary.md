# Results summary (6821 aligned faces)

## Directional leakage

|         |   cosine_gap |   knn5 |   probe_auc |
|:--------|-------------:|-------:|------------:|
| ArcFace |       0.0143 | 0.6338 |      0.9113 |
| ViT     |       0.0768 | 0.682  |      0.9224 |

## Held-out one-vs-rest AUC

|                 |   ArcFace |    ViT |
|:----------------|----------:|-------:|
| Black           |    0.9194 | 0.9528 |
| East Asian      |    0.8736 | 0.9303 |
| Indian          |    0.87   | 0.917  |
| Latino Hispanic |    0.7349 | 0.7871 |
| Middle Eastern  |    0.8251 | 0.8929 |
| Southeast Asian |    0.8598 | 0.8948 |
| White           |    0.8394 | 0.9141 |
| macro_avg       |    0.846  | 0.8984 |

## Within-race cosine similarity

|                 |   ArcFace |    ViT |
|:----------------|----------:|-------:|
| Black           |    0.0808 | 0.3383 |
| East Asian      |    0.0875 | 0.3374 |
| Indian          |    0.0783 | 0.3126 |
| Latino Hispanic |    0.077  | 0.2847 |
| Middle Eastern  |    0.072  | 0.284  |
| Southeast Asian |    0.0929 | 0.3031 |
| White           |    0.0676 | 0.2674 |

## Detector / alignment keep-rate

|                 |   keep_rate |
|:----------------|------------:|
| Black           |       0.967 |
| East Asian      |       0.974 |
| Indian          |       0.981 |
| Latino Hispanic |       0.984 |
| Middle Eastern  |       0.965 |
| Southeast Asian |       0.98  |
| White           |       0.97  |
