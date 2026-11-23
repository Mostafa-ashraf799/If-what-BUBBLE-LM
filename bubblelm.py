#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2026 Mostafa Ashraf Mostafa Eldoushy (مصطفى اشرف مصطفي الدعوشي)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
BubbleLM - verifying language-model outputs against a hierarchical "bubble tree"
memory of the training text.  (تجربة عملية لفكرة "ماذا لو" - الفقاعات)

الفكرة (زي ما اتكتبت في الكراسة):
  1) كل نص تدريب بيتقسم شجرة: مقال -> فقرات -> جمل -> توكنات (التوكن = "رابط").
     كل عقدة في الشجرة = "فقاعة".
  2) وقت الرد: احتمالات الموديل بترشّح توكنات، كل توكن بيجذب الفقاعات اللي فيها.
     لو أكتر من رابط (من السؤال + الرد) في نفس الفقاعة، احتمال إن الفقاعة دي فيها
     المعلومة المطلوبة بيزيد.
  3) قبل الإرسال: الموديل بيقارن الرد بأقوى الفقاعات (verification) ويحوّل الاحتمالات
     لإجابة موثوقة، ولو مفيش دعم كفاية بيقول "مش متأكد".
  4) scratchpad: ورقة ملاحظات بتسجل المحاولات (المرشحين + دعم كل واحد).
  5) الفقاعة الرئيسية (main bubble): اسم الموديل والشركة وقيود التخصص، بيرجعلها في كل رد.

التجربة: عالم اصطناعي من أشخاص وحقائق عنهم، موديلين عاديين (Transformer / GRU)
بيتدربوا closed-book على نصوص العالم، وبعدين نقارن:
  - الموديل العادي (greedy)
  - الموديل العادي + عتبة ثقة (أقوى baseline بسيط للهلوسة)
  - نفس الموديل + الفقاعات (مكونات الفكرة واحد واحد - ablation)
"""
import argparse
import collections
import json
import math
import random
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

# ----------------------------------------------------------------------------
# 1) العالم الاصطناعي
# ----------------------------------------------------------------------------
ATTRS = ["city", "job", "salary", "pet", "color", "hobby"]
VALUES = {
    "city": ["cairo", "giza", "luxor", "aswan", "tanta", "suez", "dubai", "riyadh", "doha", "amman",
             "beirut", "tunis", "rabat", "paris", "london", "berlin", "rome", "madrid", "tokyo",
             "seoul", "delhi", "lagos", "nairobi", "oslo", "lima"],
    "job": ["doctor", "engineer", "teacher", "pilot", "chef", "nurse", "farmer", "lawyer", "painter",
            "driver", "tailor", "baker", "banker", "guard", "editor", "singer", "dentist", "plumber",
            "joiner", "clerk", "pharmacist", "judge", "barber", "sailor", "miner"],
    "salary": [f"lvl{i}" for i in range(1, 11)],
    "pet": ["cat", "dog", "parrot", "rabbit", "turtle", "hamster", "horse", "goat", "fish", "falcon",
            "lizard", "canary", "ferret", "donkey", "snake"],
    "color": ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown",
              "gray", "teal"],
    "hobby": ["chess", "football", "swimming", "reading", "drawing", "hiking", "cooking", "fishing",
              "gaming", "cycling", "singing", "dancing", "gardening", "running", "writing",
              "photography", "climbing", "sailing", "knitting", "skating"],
}
TEMPLATES = {
    "city": ["{n} lives in {v} .", "{n} is from {v} .", "the home of {n} is {v} ."],
    "job": ["{n} works as {v} .", "{n} is a {v} .", "the job of {n} is {v} ."],
    "salary": ["{n} earns {v} .", "the pay of {n} is {v} .", "{n} gets {v} ."],
    "pet": ["{n} has a pet {v} .", "{n} keeps a {v} .", "the pet of {n} is {v} ."],
    "color": ["{n} likes the color {v} .", "the favorite color of {n} is {v} .", "{n} prefers {v} ."],
    "hobby": ["{n} enjoys {v} .", "the hobby of {n} is {v} .", "{n} spends time on {v} ."],
}
# صياغات جديدة (كلمات عمر الموديل ما شافها) لاختبار الـ paraphrase
PARAPHRASE = {"city": "{n} resides in", "job": "{n} is employed as", "salary": "{n} receives",
              "pet": "{n} owns a pet", "color": "{n} adores the color", "hobby": "{n} practices"}
# قواعد الفقاعة الرئيسية: كلمات مفتاحية لكل موضوع (برومبت قيود مكتوب بشكل بسيط)
TOPIC_KEYWORDS = {"city": ["lives", "home", "from"], "job": ["works", "job", "employed"],
                  "salary": ["earns", "pay", "gets", "receives"], "pet": ["pet", "keeps", "owns"],
                  "color": ["color", "prefers", "adores"], "hobby": ["enjoys", "hobby", "practices", "spends"]}
TRAIN_RULES = ["salary", "pet", "hobby", "none"]   # القواعد اللي الموديل العادي بيتدرب عليها
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<refuse>", "<sys>", "</sys>", "forbid", "none"]
PAD, BOS, EOS, UNK = 0, 1, 2, 3
BAD_OUT = {"<pad>", "<bos>", "<eos>", "<unk>", "<sys>", "</sys>", "forbid", "none"}

IDENTITY_TRAIN = {"model": "bubblelm", "company": "acme"}
SYL = ["ka", "ro", "mi", "vel", "su", "na", "dor", "li", "tan", "be", "zu", "fa", "ren", "mo", "ki",
       "sa", "lo", "pe", "xi", "wa", "har", "ned", "qui", "bor"]


def make_names(rng, n, taken):
    out = set()
    while len(out) < n:
        nm = "".join(rng.choice(SYL) for _ in range(rng.choice([2, 3])))
        if nm not in taken and nm not in out:
            out.add(nm)
    out = sorted(out)
    rng.shuffle(out)
    return out


def build_world(rng, n_known, n_unknown):
    taken = {w for vs in VALUES.values() for w in vs}
    for ts in TEMPLATES.values():
        for t in ts:
            taken |= set(t.split())
    taken |= {"bubblelm", "acme", "zeta", "initech", "model", "company", "name"}
    names = make_names(rng, n_known + n_unknown, taken)
    known, unknown = names[:n_known], names[n_known:]
    world = {n: {a: rng.choice(VALUES[a]) for a in ATTRS} for n in known}
    return world, known, unknown


def build_corpus(world, rng, noise=0.0):
    """docs: قايمة مقالات. المقال = فقرتين. الفقرة = 3 جمل. الجملة = (tokens, attr)."""
    docs, mentions = [], collections.Counter()
    for n, facts in world.items():
        D = rng.choices([1, 2, 3], [0.5, 0.3, 0.2])[0]
        for d in range(D):
            if d == 0:
                order = ATTRS[:]
                rng.shuffle(order)
            else:
                order = [rng.choice(ATTRS) for _ in range(6)]
            paras = []
            for pi in range(2):
                sents = []
                for a in order[pi * 3:(pi + 1) * 3]:
                    v = facts[a]
                    if noise and rng.random() < noise:
                        v = rng.choice([x for x in VALUES[a] if x != v])
                    t = rng.randrange(3)
                    sents.append((TEMPLATES[a][t].format(n=n, v=v).split(), a))
                    mentions[(n, a)] += 1
                paras.append(sents)
            docs.append(paras)
    return docs, mentions


def lm_sequences(docs, rng):
    """تسلسلات التدريب للموديل العادي: جمل منفردة + نسخ بقيود (sys prefix) + هوية."""
    seqs = []
    for doc in docs:
        for para in doc:
            for toks, attr in para:
                seqs.append(["<bos>"] + toks + ["<eos>"])
                if rng.random() < 0.35:
                    X = rng.choice(TRAIN_RULES)
                    t2 = list(toks)
                    if attr == X:
                        t2[-2] = "<refuse>"
                    seqs.append(["<bos>", "<sys>", "forbid", X, "</sys>"] + t2 + ["<eos>"])
    for _ in range(30):
        seqs.append(["<bos>", "the", "model", "name", "is", IDENTITY_TRAIN["model"], ".", "<eos>"])
        seqs.append(["<bos>", "the", "company", "name", "is", IDENTITY_TRAIN["company"], ".", "<eos>"])
    return seqs


# ----------------------------------------------------------------------------
# 2) الموديلات العادية
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(s, d, H):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.att = nn.MultiheadAttention(d, H, batch_first=True)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x, mask):
        h = s.ln1(x)
        a, _ = s.att(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + s.ff(s.ln2(x))


class TransformerLM(nn.Module):
    def __init__(s, V, d=96, L=3, H=4, maxlen=32):
        super().__init__()
        s.emb, s.pos = nn.Embedding(V, d), nn.Embedding(maxlen, d)
        s.blocks = nn.ModuleList([Block(d, H) for _ in range(L)])
        s.ln, s.out = nn.LayerNorm(d), nn.Linear(d, V)

    def forward(s, x):
        T = x.size(1)
        h = s.emb(x) + s.pos(torch.arange(T))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        for b in s.blocks:
            h = b(h, mask)
        return s.out(s.ln(h))


class GRULM(nn.Module):
    def __init__(s, V, d=128, h=192, L=2):
        super().__init__()
        s.emb = nn.Embedding(V, d)
        s.rnn = nn.GRU(d, h, num_layers=L, batch_first=True)
        s.out = nn.Linear(h, V)

    def forward(s, x):
        o, _ = s.rnn(s.emb(x))
        return s.out(o)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def train_lm(model, seqs, stoi, epochs, lr, bs=128, seed=0, log=None):
    torch.manual_seed(seed)
    N = len(seqs)
    Lmax = max(len(s) for s in seqs)
    X = torch.zeros(N, Lmax, dtype=torch.long)
    lens = torch.tensor([len(s) for s in seqs])
    for i, s in enumerate(seqs):
        X[i, :len(s)] = torch.tensor([stoi[w] for w in s])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    spe = math.ceil(N / bs)
    total = epochs * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.1)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot, cnt = 0.0, 0
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            ml = int(lens[idx].max())
            xb = X[idx, :ml]
            logits = model(xb[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), xb[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            cnt += 1
        if log and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            log(f"    epoch {ep + 1}/{epochs} loss={tot / cnt:.4f}")
    model.eval()
    return model


class LM:
    def __init__(self, model, itos):
        self.model = model.eval()
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.bad = torch.tensor([self.stoi[w] for w in BAD_OUT])

    def id(self, w):
        return self.stoi.get(w, UNK)

    @torch.no_grad()
    def next_logp(self, seqs):
        outs = []
        for c in range(0, len(seqs), 1024):
            chunk = seqs[c:c + 1024]
            L = max(len(s) for s in chunk)
            x = torch.zeros(len(chunk), L, dtype=torch.long)
            for i, s in enumerate(chunk):
                x[i, :len(s)] = torch.tensor(s)
            logits = self.model(x)
            idx = torch.tensor([len(s) - 1 for s in chunk])
            last = logits[torch.arange(len(chunk)), idx]
            outs.append(F.log_softmax(last, -1))
        return torch.cat(outs, 0)


# ----------------------------------------------------------------------------
# 3) شجرة الفقاعات (الفهرس)
# ----------------------------------------------------------------------------
DISC = [1.0, 0.9, 0.8]       # خصم المستويات: جملة / فقرة / مقال
LEVELS = {"sent": [0], "par": [1], "doc": [2], "tree": [0, 1, 2]}


class BubbleIndex:
    def __init__(self, docs, power=2.0, stop_ratio=0.25):
        self.mem = [[], [], []]
        self.inv = [collections.defaultdict(list) for _ in range(3)]
        df = collections.Counter()
        ns = 0
        for doc in docs:
            dset = set()
            for para in doc:
                pset = set()
                for toks, _a in para:
                    s = set(toks)
                    sid = len(self.mem[0])
                    self.mem[0].append(s)
                    for t in s:
                        self.inv[0][t].append(sid)
                        df[t] += 1
                    pset |= s
                    ns += 1
                pid = len(self.mem[1])
                self.mem[1].append(pset)
                for t in pset:
                    self.inv[1][t].append(pid)
                dset |= pset
            did = len(self.mem[2])
            self.mem[2].append(dset)
            for t in dset:
                self.inv[2][t].append(did)
        self.df, self.ns = df, ns
        self.power, self.stop_ratio = power, stop_ratio
        self.idf_unseen = math.log(1 + ns / 0.5)

    def weight(self, t):
        d = self.df.get(t)
        if d is None:                      # توكن عمر الفهرس ما شافه = أقوى دليل (ومفيش فقاعة فيها)
            return self.idf_unseen ** self.power
        if d / self.ns > self.stop_ratio:  # كلمات شايعة (is, ".") مش دليل
            return 0.0
        return math.log(1 + self.ns / d) ** self.power

    def activation(self, ctx):
        w = {}
        for t in set(ctx):
            x = self.weight(t)
            if x > 0:
                w[t] = x
        acc = [collections.defaultdict(float) for _ in range(3)]
        for t, x in w.items():
            for lv in range(3):
                for b in self.inv[lv].get(t, ()):
                    acc[lv][b] += x
        return acc, sum(w.values())

    def cand_support(self, acc, tot, cand, levels):
        """لو المرشح ده اتضاف للسياق: أعلى نسبة تغطية في أي فقاعة (هو "جذب الفقاعة")."""
        wc = self.weight(cand)
        if wc == 0:
            return 0.0
        best = 0.0
        for lv in levels:
            disc = DISC[lv] if len(levels) > 1 else 1.0
            a = acc[lv]
            for b in self.inv[lv].get(cand, ()):
                cov = disc * (a.get(b, 0.0) + wc) / (tot + wc)
                if cov > best:
                    best = cov
        return best

    def _sset(self, t):
        c = self.__dict__.setdefault("_sc", {})
        if t not in c:
            c[t] = set(self.inv[0].get(t, ()))
        return c[t]

    def relation_ok(self, prompt, ans, min_df=30):
        """فحص العلاقة: كلمات العلاقة في السؤال (متكررة لكن مش شايعة، يعني مش اسم ومش stopword)
        لازم تظهر في نفس جملة-فقاعة مع واحدة من كلمات الإجابة. لو مفيش كلمات علاقة: مفيش قيد."""
        rel = [t for t in set(prompt) if min_df <= self.df.get(t, 0) <= self.stop_ratio * self.ns]
        if not rel:
            return True
        return any(self._sset(r) & self._sset(a) for r in rel for a in ans)

    def score(self, ev, level, prompt=None, ans=None, rel=False):
        """verification: أعلى نسبة تغطية للدليل (سؤال + رد) داخل فقاعة واحدة."""
        acc, tot = self.activation(ev)
        if tot == 0:
            return 0.0
        levels = LEVELS[level]
        best = 0.0
        for lv in levels:
            if acc[lv]:
                disc = DISC[lv] if len(levels) > 1 else 1.0
                best = max(best, disc * max(acc[lv].values()) / tot)
        if rel and prompt is not None and not self.relation_ok(prompt, ans):
            return 0.0
        return best


# ----------------------------------------------------------------------------
# 4) الفقاعة الرئيسية
# ----------------------------------------------------------------------------
class MainBubble:
    def __init__(self, model_name, company, forbidden=()):
        self.identity = {"model": model_name, "company": company}
        self.forbidden = set(forbidden)

    def consult(self, prompt):
        """بيتراجع في كل رد: هوية؟ موضوع ممنوع؟ غير كده None."""
        p = tuple(prompt)
        if p[:2] == ("the", "model") and "name" in p:
            return ("answer", self.identity["model"])
        if p[:2] == ("the", "company") and "name" in p:
            return ("answer", self.identity["company"])
        for topic in self.forbidden:
            if any(k in p for k in TOPIC_KEYWORDS[topic]):
                return ("refuse", topic)
        return None


# ----------------------------------------------------------------------------
# 5) فك التشفير: عادي / beam + نودج الفقاعات + verification + scratchpad
# ----------------------------------------------------------------------------
@dataclass
class Cfg:
    B: int = 1            # beam width
    K: int = 1            # مرشحين لكل خطوة
    lam: float = 0.0      # قوة نودج الفقاعات على الاحتمالات
    mu: float = 0.0       # وزن الـ verification في اختيار الرد النهائي (rerank)
    level: str = "tree"   # sent | par | doc | tree
    verify: bool = False
    rel: bool = False     # فحص العلاقة بين السؤال والإجابة داخل الفقاعات
    maxlen: int = 3


def decode_batch(lm, prompts, index, cfg, prefixes=None):
    Q = len(prompts)
    prefixes = prefixes or [[] for _ in range(Q)]
    base = [[BOS] + [lm.id(w) for w in (prefixes[q] + prompts[q])] for q in range(Q)]
    beams = [[dict(s=0.0, v=0.0, w=[], done=False)] for _ in range(Q)]
    levels = LEVELS[cfg.level]
    use_nudge = cfg.lam > 0 and index is not None
    cache = {}
    for _step in range(cfg.maxlen):
        reqs = [(q, i) for q in range(Q) for i, b in enumerate(beams[q]) if not b["done"]]
        if not reqs:
            break
        seqs = [base[q] + [lm.id(x) for x in beams[q][i]["w"]] for q, i in reqs]
        lp = lm.next_logp(seqs)
        lp[:, lm.bad] = -1e9
        topv, topi = lp.topk(cfg.K, dim=-1)
        topv, topi = topv.tolist(), topi.tolist()
        new = [[b for b in beams[q] if b["done"]] for q in range(Q)]
        for r, (q, i) in enumerate(reqs):
            b = beams[q][i]
            if use_nudge:
                key = (q, tuple(b["w"]))
                if key not in cache:
                    cache[key] = index.activation(prompts[q] + b["w"])
                acc, tot = cache[key]
            for v, ti in zip(topv[r], topi[r]):
                w = lm.itos[ti]
                if w == ".":
                    if not b["w"]:
                        continue
                    new[q].append(dict(s=b["s"] + v, v=b["v"], w=b["w"], done=True))
                else:
                    s = b["s"] + v
                    if use_nudge:
                        s += cfg.lam * index.cand_support(acc, tot, w, levels)
                    new[q].append(dict(s=s, v=b["v"] + v, w=b["w"] + [w], done=False))
        for q in range(Q):
            new[q].sort(key=lambda x: -x["s"])
            beams[q] = new[q][:cfg.B]
    out = []
    for q in range(Q):
        cands = [b for b in beams[q] if b["w"]]
        if not cands:
            out.append(dict(ans=None, conf=0.0, ver=None, scratch=[]))
            continue
        for b in cands:
            b["ver"] = index.score(prompts[q] + b["w"], cfg.level, prompts[q], b["w"], cfg.rel) if cfg.verify else None
        best = max(cands, key=lambda b: b["s"] + (cfg.mu * b["ver"] if cfg.verify else 0.0))
        scratch = [dict(reply=" ".join(b["w"]), lm_logp=round(b["s"], 3),
                        verif=None if b["ver"] is None else round(b["ver"], 3),
                        chosen=b is best) for b in sorted(cands, key=lambda x: -x["s"])]
        out.append(dict(ans=" ".join(best["w"]), conf=math.exp(best["v"]), ver=best["ver"], scratch=scratch))
    return out


def prompt_for(n, a, paraphrase=False):
    tpl = PARAPHRASE[a] if paraphrase else TEMPLATES[a][0].split("{v}")[0]
    return tpl.format(n=n).split()


# ----------------------------------------------------------------------------
# 6) التقييم
# ----------------------------------------------------------------------------
def make_queries(entities, unknown, world, mentions, rng):
    qs = []
    for n in entities:
        for a in ATTRS:
            qs.append(dict(kind="known", n=n, a=a, truth=world[n][a], freq=mentions[(n, a)]))
    for n in unknown:
        for a in rng.sample(ATTRS, 2):
            qs.append(dict(kind="unknown", n=n, a=a, truth=None, freq=0))
    return qs


def run_system(lm, index, qs, cfg, score_kind, paraphrase=False):
    prompts = [prompt_for(q["n"], q["a"], paraphrase) for q in qs]
    res = decode_batch(lm, prompts, index, cfg)
    ans, score = [], []
    for r in res:
        a = r["ans"]
        if a == "<refuse>":
            a = None
        ans.append(a)
        if score_kind == "conf":
            score.append(r["conf"])
        else:
            score.append(-1.0 if r["ver"] is None else r["ver"])
    return ans, np.array(score), res


def arrays(qs, ans):
    unknown = np.array([q["kind"] == "unknown" for q in qs])
    correct = np.array([(a is not None and q["truth"] is not None and a == q["truth"]) for q, a in zip(qs, ans)])
    present = np.array([a is not None for a in ans])
    return unknown, correct, present


def utility(answered, correct, unknown, penalty=1.0):
    good = (answered & correct).sum() + ((~answered) & unknown).sum()
    bad = (answered & ~correct).sum()
    return good - penalty * bad


def tune_threshold(score, correct, unknown, present):
    cands = sorted(set(score[present].tolist()))
    best_u, best_t = -1e18, float("inf")
    for t in [float("-inf")] + cands:
        ans = present & (score >= t)
        u = utility(ans, correct, unknown)
        if u > best_u + 1e-9:
            best_u, best_t = u, t
    return best_t


def metrics(score, correct, unknown, present, tau, freq):
    ans = present & (score >= tau)
    known = ~unknown
    nk, nu = known.sum(), unknown.sum()
    c = (ans & correct).sum()
    wrong_known = (ans & known & ~correct).sum()
    hall_unknown = (ans & unknown).sum()
    abst_unknown = ((~ans) & unknown).sum()
    n_ans = ans.sum()
    rare = known & (freq == 1)
    freqm = known & (freq >= 2)
    return dict(
        recall=float(c / nk),
        precision=float(c / n_ans) if n_ans else 1.0,
        hall_rate=float((wrong_known + hall_unknown) / len(score)),   # هلوسة / كل الأسئلة
        wrong_known=int(wrong_known), hall_unknown=int(hall_unknown), abst_unknown=int(abst_unknown),
        unknown_total=int(nu), known_total=int(nk),
        rare_recall=float((ans & correct & rare).sum() / max(1, rare.sum())),
        freq_recall=float((ans & correct & freqm).sum() / max(1, freqm.sum())),
        utility=float(utility(ans, correct, unknown)),
        coverage=float(n_ans / len(score)),
    )


def recall_at_precision(score, correct, unknown, present, target):
    """أعلى recall بحيث precision >= target (بتحسب على التست مباشرة - أفضل حالة لكل نظام)."""
    nk = (~unknown).sum()
    best = 0.0
    for t in sorted(set(score[present].tolist())):
        ans = present & (score >= t)
        n = ans.sum()
        if n and (ans & correct).sum() / n >= target:
            best = max(best, (ans & correct).sum() / nk)
    return float(best)


# ----------------------------------------------------------------------------
# 7) اختبارات الفقاعة الرئيسية
# ----------------------------------------------------------------------------
def rule_test(lm, index, test_entities, world, rng, bubble_cfg):
    sample = rng.sample(test_entities, min(120, len(test_entities)))
    out = {}
    for R, kind in [("salary", "seen"), ("pet", "seen"), ("color", "UNSEEN")]:
        qs = [(n, a) for n in sample for a in ATTRS]
        prompts = [prompt_for(n, a) for n, a in qs]
        # الموديل العادي: القاعدة في برومبت sys زي ما اتدرب
        pre = [["<sys>", "forbid", R, "</sys>"] for _ in qs]
        base = decode_batch(lm, prompts, None, Cfg(), prefixes=pre)
        # BubbleLM: القاعدة في الفقاعة الرئيسية (من غير إعادة تدريب)
        main = MainBubble("bubblelm", "acme", forbidden=[R])
        bub_ref = []
        rest_idx, rest_prompts = [], []
        for i, p in enumerate(prompts):
            m = main.consult(p)
            if m and m[0] == "refuse":
                bub_ref.append(True)
            else:
                bub_ref.append(False)
                rest_idx.append(i)
                rest_prompts.append(p)
        stats = {}
        for name, refused in [("transformer", [r["ans"] == "<refuse>" for r in base]), ("bubble", bub_ref)]:
            forb = [a == R for _n, a in qs]
            ref_forb = sum(r for r, f in zip(refused, forb) if f) / max(1, sum(forb))
            ref_ok = sum(r for r, f in zip(refused, forb) if not f) / max(1, len(forb) - sum(forb))
            stats[name] = dict(refuse_forbidden=ref_forb, over_refuse_allowed=ref_ok)
        out[f"{R} ({kind} rule)"] = stats
    return out


def identity_test(lm):
    res = {}
    phrases = {"model": ["the", "model", "name", "is"], "company": ["the", "company", "name", "is"]}
    base = decode_batch(lm, list(phrases.values()), None, Cfg())
    for (k, _p), r in zip(phrases.items(), base):
        res[k] = r["ans"]
    return res


# ----------------------------------------------------------------------------
# 8) التشغيل
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=400)
    ap.add_argument("--unknown", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--no_gru", action="store_true")
    ap.add_argument("--out", type=str, default="results.json")
    args = ap.parse_args()
    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:6.1f}s] {s}", flush=True)

    rng = random.Random(args.seed)
    world, known, unknown = build_world(rng, args.entities, args.unknown)
    docs, mentions = build_corpus(world, rng, noise=args.noise)
    seqs = lm_sequences(docs, rng)
    toks = sorted({t for s in seqs for t in s} - set(SPECIALS))
    itos = SPECIALS + toks
    stoi = {w: i for i, w in enumerate(itos)}
    V = len(itos)
    n_sents = sum(len(p) for d in docs for p in d)
    log(f"world: {len(world)} entities, {len(docs)} docs, {n_sents} sentences, {len(seqs)} training seqs, vocab={V}")
    mc = collections.Counter(mentions.values())
    log(f"fact mention counts: {dict(sorted(mc.items()))}")

    # تقسيم val/test بالكيانات
    ents = known[:]
    rng.shuffle(ents)
    nval = int(0.3 * len(ents))
    val_e, test_e = ents[:nval], ents[nval:]
    val_u, test_u = unknown[:len(unknown) // 3], unknown[len(unknown) // 3:]
    qs_val = make_queries(val_e, val_u, world, mentions, rng)
    qs_test = make_queries(test_e, test_u, world, mentions, rng)
    log(f"val queries={len(qs_val)} test queries={len(qs_test)}")

    index = BubbleIndex(docs, power=2.0)
    log(f"bubble index: sentence-bubbles={index.ns}, paragraph-bubbles={len(index.mem[1])}, doc-bubbles={len(index.mem[2])}")

    models = {}
    torch.manual_seed(args.seed)
    tf = TransformerLM(V)
    log(f"training Transformer ({nparams(tf):,} params), epochs={args.epochs}")
    train_lm(tf, seqs, stoi, args.epochs, 3e-3, seed=args.seed, log=log)
    models["Transformer"] = LM(tf, itos)
    if not args.no_gru:
        gru = GRULM(V)
        log(f"training GRU ({nparams(gru):,} params), epochs={args.epochs}")
        train_lm(gru, seqs, stoi, args.epochs, 4e-3, seed=args.seed, log=log)
        models["GRU"] = LM(gru, itos)

    rows = []   # كل صف = نظام

    def eval_row(label, lm, cfg, score_kind, abstain):
        a_v, s_v, _ = run_system(lm, index, qs_val, cfg, score_kind)
        a_t, s_t, res_t = run_system(lm, index, qs_test, cfg, score_kind)
        uv, cv, pv = arrays(qs_val, a_v)
        ut, ct, pt = arrays(qs_test, a_t)
        tau = tune_threshold(s_v, cv, uv, pv) if abstain else float("-inf")
        freq_t = np.array([q["freq"] for q in qs_test])
        m = metrics(s_t, ct, ut, pt, tau, freq_t)
        m["R@P95"] = recall_at_precision(s_t, ct, ut, pt, 0.95)
        m["R@P99"] = recall_at_precision(s_t, ct, ut, pt, 0.99)
        m["tau"] = None if tau == float("-inf") else float(tau)
        m["label"] = label
        rows.append(m)
        log(f"  {label:55s} recall={m['recall']:.3f} prec={m['precision']:.3f} hall={m['hall_rate']:.3f} "
            f"U={m['utility']:.0f} R@P99={m['R@P99']:.3f}")
        return res_t

    scratch_demo = None
    for mname, lm in models.items():
        log(f"evaluating {mname}")
        eval_row(f"{mname} (greedy)", lm, Cfg(), "conf", False)
        eval_row(f"{mname} + confidence threshold", lm, Cfg(), "conf", True)
        if mname == "Transformer":
            eval_row("  bubble: verify only - doc level", lm, Cfg(B=5, K=6, verify=True, level="doc"), "ver", True)
            eval_row("  bubble: verify only - paragraph level", lm, Cfg(B=5, K=6, verify=True, level="par"), "ver", True)
            eval_row("  bubble: verify only - sentence level", lm, Cfg(B=5, K=6, verify=True, level="sent"), "ver", True)
            eval_row("  bubble: verify only - tree (all levels)", lm, Cfg(B=5, K=6, verify=True, level="tree"), "ver", True)
            eval_row("  bubble: verify + rerank (tree)", lm, Cfg(B=5, K=6, mu=8.0, verify=True, level="tree"), "ver", True)
            res = eval_row("  bubble: FULL nudge+rerank+verify (tree)", lm,
                           Cfg(B=5, K=6, lam=2.0, mu=8.0, verify=True, level="tree"), "ver", True)
            eval_row("  bubble: FULL (sentence level only)", lm,
                     Cfg(B=5, K=6, lam=2.0, mu=8.0, verify=True, level="sent"), "ver", True)
            eval_row("  bubble: FULL + relation check (tree)", lm,
                     Cfg(B=5, K=6, lam=2.0, mu=8.0, verify=True, rel=True, level="tree"), "ver", True)
            scratch_demo = [(q, r) for q, r in zip(qs_test, res)][:0]
            # عينات scratchpad: واحد صح، واحد اتصلح، واحد رفض (unknown)
            picks = []
            for q, r in zip(qs_test, res):
                if q["kind"] == "unknown" and len([p for p in picks if p[0]["kind"] == "unknown"]) < 2:
                    picks.append((q, r))
            for q, r in zip(qs_test, res):
                if q["kind"] == "known" and r["ans"] != q["truth"] and len(
                        [p for p in picks if p[0]["kind"] == "known"]) < 2 and any(
                        s["reply"] == q["truth"] for s in r["scratch"]):
                    picks.append((q, r))
            scratch_demo = picks
        else:
            eval_row(f"  bubble over {mname}: FULL (tree)", lm,
                     Cfg(B=5, K=6, lam=2.0, mu=8.0, verify=True, level="tree"), "ver", True)

    # حساسية: idf power
    index.power = 1.0
    eval_row("  [ablation] FULL with idf power=1", models["Transformer"],
             Cfg(B=5, K=6, lam=2.0, mu=8.0, verify=True, level="tree"), "ver", True)
    index.power = 2.0

    # Paraphrase (كلمات جديدة في السؤال)
    log("paraphrase test")
    sample_q = [q for q in qs_test if q["kind"] == "known"]
    rng.shuffle(sample_q)
    sample_q = sample_q[:300]
    para = {}
    lm = models["Transformer"]
    for label, cfg, kind in [("Transformer (greedy)", Cfg(), "conf"),
                             ("bubble FULL (tree)", Cfg(B=5, K=6, lam=2.0, mu=8.0, verify=True, level="tree"), "ver")]:
        a, s, _ = run_system(lm, index, sample_q, cfg, kind, paraphrase=True)
        u, c, p = arrays(sample_q, a)
        # عتبة من نفس الصف الرئيسي
        tau = [r["tau"] for r in rows if r["label"].strip().startswith("bubble: FULL nudge")][0] if kind == "ver" else None
        ans = p & (s >= (tau if tau is not None else float("-inf")))
        para[label] = dict(recall=float((ans & c).sum() / len(sample_q)),
                           answered=float(ans.mean()), wrong=int((ans & ~c).sum()))
        log(f"  {label}: {para[label]}")

    # الفقاعة الرئيسية
    log("main-bubble tests")
    rules = rule_test(models["Transformer"], index, test_e, world, rng, None)
    for k, v in rules.items():
        log(f"  rule {k}: {v}")
    ident_base = identity_test(models["Transformer"])
    ident = dict(transformer_answers=ident_base,
                 bubble_before_change=MainBubble("bubblelm", "acme").identity,
                 bubble_after_change=MainBubble("zeta", "initech").identity)
    log(f"  identity: {ident}")

    # أخطاء "verified-but-wrong" (false positives) في الصف الكامل
    full_label = "bubble: FULL nudge+rerank+verify (tree)"
    tau_full = [r["tau"] for r in rows if r["label"].strip() == full_label][0]
    a_t, s_t, res_t = run_system(models["Transformer"], index, qs_test,
                                 Cfg(B=5, K=6, lam=2.0, mu=8.0, verify=True, level="tree"), "ver")
    fp = []
    for q, a, s in zip(qs_test, a_t, s_t):
        if a is not None and s >= tau_full and a != q["truth"]:
            fp.append(dict(kind=q["kind"], n=q["n"], attr=q["a"], truth=q["truth"], answer=a, verif=round(float(s), 3)))
    log(f"  verified-but-wrong answers (FULL): {len(fp)}")

    out = dict(args=vars(args), rows=rows, paraphrase=para, rules=rules, identity=ident,
               verified_wrong=fp[:25], n_verified_wrong=len(fp),
               scratch_demo=[dict(q=q, scratch=r["scratch"], ans=r["ans"], ver=r["ver"]) for q, r in (scratch_demo or [])],
               fact_mentions={str(k): v for k, v in mc.items()},
               n_params=dict(transformer=nparams(tf), gru=None if args.no_gru else nparams(models["GRU"].model)))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=float)
    log(f"saved {args.out}")


if __name__ == "__main__":
    main()
