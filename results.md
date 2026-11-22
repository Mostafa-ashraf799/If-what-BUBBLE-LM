### Main experiment (mean ± std over 3 seeds, clean data)

| System | Recall | Precision | Hallucination rate | Utility | Rare-fact recall | Freq-fact recall | Recall@P99 |
|---|---|---|---|---|---|---|---|
| Transformer (greedy) | 0.816 ±0.008 | 0.729 ±0.008 | 0.271 ±0.008 | 862 ±29 | 0.739 ±0.014 | 0.945 ±0.004 | 0.506 ±0.015 |
| Transformer + confidence threshold | 0.723 ±0.024 | 0.864 ±0.018 | 0.102 ±0.016 | 1131 ±60 | 0.623 ±0.033 | 0.891 ±0.011 | 0.506 ±0.015 |
| bubble: verify only - doc level | 0.815 ±0.009 | 0.996 ±0.002 | 0.003 ±0.001 | 1565 ±13 | 0.739 ±0.014 | 0.943 ±0.005 | 0.816 ±0.008 |
| bubble: verify only - paragraph level | 0.815 ±0.009 | 0.996 ±0.002 | 0.003 ±0.001 | 1565 ±13 | 0.739 ±0.014 | 0.943 ±0.005 | 0.816 ±0.008 |
| bubble: verify only - sentence level | 0.815 ±0.009 | 0.996 ±0.002 | 0.003 ±0.001 | 1565 ±13 | 0.739 ±0.014 | 0.943 ±0.005 | 0.816 ±0.008 |
| bubble: verify only - tree (all levels) | 0.815 ±0.009 | 0.996 ±0.002 | 0.003 ±0.001 | 1565 ±13 | 0.739 ±0.014 | 0.943 ±0.005 | 0.816 ±0.008 |
| bubble: verify + rerank (tree) | 0.881 ±0.002 | 0.992 ±0.004 | 0.007 ±0.003 | 1667 ±4 | 0.826 ±0.003 | 0.971 ±0.003 | 0.849 ±0.044 |
| bubble: FULL nudge+rerank+verify (tree) | 0.899 ±0.003 | 0.990 ±0.006 | 0.009 ±0.005 | 1695 ±10 | 0.853 ±0.007 | 0.976 ±0.003 | 0.786 ±0.161 |
| bubble: FULL (sentence level only) | 0.899 ±0.003 | 0.990 ±0.006 | 0.009 ±0.005 | 1695 ±10 | 0.853 ±0.007 | 0.976 ±0.003 | 0.786 ±0.161 |
| bubble: FULL + relation check (tree) | 0.902 ±0.003 | 1.000 ±0.001 | 0.000 ±0.001 | 1715 ±5 | 0.858 ±0.006 | 0.977 ±0.002 | 0.903 ±0.003 |
| GRU (greedy) | 0.654 ±0.019 | 0.585 ±0.017 | 0.415 ±0.017 | 318 ±64 | 0.550 ±0.013 | 0.828 ±0.022 | 0.378 ±0.084 |
| GRU + confidence threshold | 0.490 ±0.015 | 0.967 ±0.020 | 0.015 ±0.010 | 994 ±23 | 0.361 ±0.023 | 0.706 ±0.017 | 0.378 ±0.084 |
| bubble over GRU: FULL (tree) | 0.786 ±0.015 | 0.991 ±0.003 | 0.006 ±0.002 | 1509 ±28 | 0.717 ±0.017 | 0.902 ±0.009 | 0.729 ±0.097 |
| [ablation] FULL with idf power=1 | 0.850 ±0.002 | 0.965 ±0.006 | 0.027 ±0.005 | 1577 ±9 | 0.787 ±0.007 | 0.956 ±0.009 | 0.676 ±0.072 |

### Noisy corpus (10% of the facts written wrong in the training text, seed 0)

| System | Recall | Precision | Hallucination rate | Utility | Rare-fact recall | Freq-fact recall | Recall@P99 |
|---|---|---|---|---|---|---|---|
| Transformer (greedy) | 0.733 | 0.655 | 0.345 | 582 | 0.648 | 0.868 | 0.008 |
| Transformer + confidence threshold | 0.657 | 0.809 | 0.139 | 1017 | 0.573 | 0.792 | 0.008 |
| bubble: verify only - doc level | 0.726 | 0.899 | 0.073 | 1282 | 0.643 | 0.857 | 0.000 |
| bubble: verify only - paragraph level | 0.726 | 0.899 | 0.073 | 1282 | 0.643 | 0.857 | 0.000 |
| bubble: verify only - sentence level | 0.726 | 0.899 | 0.073 | 1282 | 0.643 | 0.857 | 0.000 |
| bubble: verify only - tree (all levels) | 0.726 | 0.899 | 0.073 | 1282 | 0.643 | 0.857 | 0.000 |
| bubble: verify + rerank (tree) | 0.808 | 0.900 | 0.080 | 1407 | 0.761 | 0.884 | 0.000 |
| bubble: FULL nudge+rerank+verify (tree) | 0.826 | 0.897 | 0.085 | 1428 | 0.787 | 0.887 | 0.000 |
| bubble: FULL (sentence level only) | 0.826 | 0.897 | 0.085 | 1428 | 0.787 | 0.887 | 0.000 |
| bubble: FULL + relation check (tree) | 0.827 | 0.902 | 0.080 | 1438 | 0.788 | 0.888 | 0.000 |
| GRU (greedy) | 0.593 | 0.530 | 0.470 | 114 | 0.498 | 0.747 | 0.005 |
| GRU + confidence threshold | 0.485 | 0.830 | 0.089 | 848 | 0.384 | 0.648 | 0.005 |
| bubble over GRU: FULL (tree) | 0.708 | 0.884 | 0.083 | 1234 | 0.645 | 0.809 | 0.000 |
| [ablation] FULL with idf power=1 | 0.768 | 0.878 | 0.096 | 1311 | 0.713 | 0.857 | 0.000 |

### Paraphrase test (question uses words never seen in training)
- seed 0: Transformer (greedy): recall=0.37, answered=0.89, wrong=156; bubble FULL (tree): recall=0.00, answered=0.00, wrong=0
- seed 1: Transformer (greedy): recall=0.40, answered=0.58, wrong=55; bubble FULL (tree): recall=0.00, answered=0.00, wrong=0
- seed 2: Transformer (greedy): recall=0.38, answered=0.84, wrong=138; bubble FULL (tree): recall=0.00, answered=0.00, wrong=0

### Main bubble (restriction rules + identity)
- salary (seen rule): transformer: refuses forbidden=1.00, over-refuses allowed=0.004 | bubble: refuses forbidden=1.00, over-refuses allowed=0.000
- pet (seen rule): transformer: refuses forbidden=0.99, over-refuses allowed=0.002 | bubble: refuses forbidden=1.00, over-refuses allowed=0.000
- color (UNSEEN rule): transformer: refuses forbidden=0.00, over-refuses allowed=0.021 | bubble: refuses forbidden=1.00, over-refuses allowed=0.000
- identity (plain Transformer after training): {'model': 'bubblelm', 'company': 'acme'}
- identity (bubble, rename without retraining): {'model': 'zeta', 'company': 'initech'}

### Params: {"transformer": 444741, "gru": 584037}
### Verified-but-wrong counts (FULL, before relation check): [10, 8, 30]  noisy: 159