# BubbleLM

Verifying language-model outputs against a hierarchical **bubble tree** memory of the training text.
Every training document is split into a tree (document → paragraph → sentence → token). At generation
time the model's likely tokens activate the bubbles they belong to, and each draft reply is verified
against the strongest bubbles before it is sent. A scratchpad records the candidates, and a *main
bubble* holds the model's identity and restriction rules.

**Author:** Mostafa Ashraf Mostafa Eldoushy (مصطفى اشرف مصطفي الدعوشي)
**License:** Apache License 2.0 (see `LICENSE` and `NOTICE`)

## Files

| File | Purpose |
|---|---|
| `bubblelm.py` | Full implementation and experiment (bubble tree, activation, verification, relation check, scratchpad, main bubble, Transformer/GRU baselines) |
| `aggregate.py` | Aggregates the per-seed JSON results into `results.md` |
| `results.md` | Result tables (3 seeds, clean data; plus a noisy-corpus run) |

## Run

```bash
pip install torch numpy
python3 bubblelm.py --epochs 80 --seed 0 --out v2_s0.json   # one seed, ~7 min on 2 CPU cores
python3 bubblelm.py --epochs 80 --seed 0 --noise 0.1 --out v2_noise.json   # 10% corrupted training text
python3 aggregate.py                                          # needs v2_s0/1/2.json and v2_noise.json
```

## Limits (see the paper)

The experiment runs on a synthetic world with exact-token matching. It abstains on paraphrased
questions, it cannot detect facts that are wrong in its own training text, and the tree levels showed no
difference in this world.
