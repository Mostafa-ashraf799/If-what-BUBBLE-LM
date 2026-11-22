import json, sys, numpy as np

seeds = [json.load(open(f"v2_s{s}.json")) for s in (0, 1, 2)]
noise = json.load(open("v2_noise.json"))
keys = ["recall", "precision", "hall_rate", "utility", "rare_recall", "freq_recall", "R@P99"]


def table(runs, title):
    labels = [r["label"].strip() for r in runs[0]["rows"]]
    out = [f"### {title}", "",
           "| System | Recall | Precision | Hallucination rate | Utility | Rare-fact recall | Freq-fact recall | Recall@P99 |",
           "|---|---|---|---|---|---|---|---|"]
    for i, lab in enumerate(labels):
        vals = []
        for k in keys:
            xs = np.array([run["rows"][i][k] for run in runs])
            f = (lambda v: f"{v:.0f}") if k == "utility" else (lambda v: f"{v:.3f}")
            vals.append(f(xs.mean()) + ("" if len(runs) == 1 else f" ±{xs.std():.3f}" if k != "utility" else f" ±{xs.std():.0f}"))
        out.append(f"| {lab} | " + " | ".join(vals) + " |")
    return "\n".join(out)


txt = [table(seeds, "Main experiment (mean ± std over 3 seeds, clean data)"), "",
       table([noise], "Noisy corpus (10% of the facts written wrong in the training text, seed 0)"), ""]

txt.append("### Paraphrase test (question uses words never seen in training)")
for s, r in zip((0, 1, 2), seeds):
    txt.append(f"- seed {s}: " + "; ".join(f"{k}: recall={v['recall']:.2f}, answered={v['answered']:.2f}, wrong={v['wrong']}" for k, v in r["paraphrase"].items()))
txt.append("")
txt.append("### Main bubble (restriction rules + identity)")
for k in seeds[0]["rules"]:
    row = []
    for name in ("transformer", "bubble"):
        a = [r["rules"][k][name]["refuse_forbidden"] for r in seeds]
        b = [r["rules"][k][name]["over_refuse_allowed"] for r in seeds]
        row.append(f"{name}: refuses forbidden={np.mean(a):.2f}, over-refuses allowed={np.mean(b):.3f}")
    txt.append(f"- {k}: " + " | ".join(row))
txt.append(f"- identity (plain Transformer after training): {seeds[0]['identity']['transformer_answers']}")
txt.append(f"- identity (bubble, rename without retraining): {seeds[0]['identity']['bubble_after_change']}")
txt.append("")
txt.append("### Params: " + json.dumps(seeds[0]["n_params"]))
txt.append("### Verified-but-wrong counts (FULL, before relation check): " + str([r["n_verified_wrong"] for r in seeds]) + f"  noisy: {noise['n_verified_wrong']}")
open("results.md", "w").write("\n".join(txt))
print("\n".join(txt))
