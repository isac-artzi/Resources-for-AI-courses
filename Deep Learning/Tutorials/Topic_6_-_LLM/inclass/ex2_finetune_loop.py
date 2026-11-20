"""
AIT-204 · Topic 6 · In-class Exercise 2 — Write the loop the Trainer runs for you
Time: ~15–20 min (about 20 s of CPU when it runs)   |   Needs: torch

    1. collate      : pad a batch → input_ids + attention_mask         (what a tokenizer returns)
    2. schedule     : linear warmup, then cosine decay                  (warmup_ratio in TrainingArguments)
    3. train_step   : forward → loss → backward → clip → step → zero    (the inside of trainer.train())
    4. experiments  : three learning rates + a from-scratch baseline, one row per run

There is nothing to download. The script first PRE-TRAINS a tiny causal transformer on unlabeled
sequences from two made-up "dialects" (next-token prediction, exactly CLM). You then FINE-TUNE it to
tell the dialects apart from only 32 labeled examples. Both dialects use every symbol equally often;
they differ only in which symbol tends to follow which, so the model has to know the language to win.

Fill in every "TODO".
"""
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, K = 0, 12              # id 0 pads; symbols are ids 1…12
V, D = K + 1, 32            # vocabulary size, model width


# ---- Given: the data ------------------------------------------------------
def sample(dialect, n, g, lo=10, hi=24):
    """n sequences of random length. Dialect 0 usually hops +1 around the circle of symbols, dialect 1 hops +5."""
    hop, seqs = (1 if dialect == 0 else 5), []
    for _ in range(n):
        T = int(torch.randint(lo, hi + 1, (1,), generator=g))
        cur = int(torch.randint(1, K + 1, (1,), generator=g))
        s = [cur]
        for _ in range(T - 1):
            cur = (cur - 1 + hop) % K + 1 if torch.rand(1, generator=g).item() < 0.6 else int(torch.randint(1, K + 1, (1,), generator=g))
            s.append(cur)
        seqs.append(s)
    return seqs


# ---------------------------------------------------------------------------
# TODO 1 — collate: what the tokenizer hands the model
# ---------------------------------------------------------------------------
def collate(seqs):
    """seqs: list of lists of ids, different lengths. Returns (input_ids, attention_mask), both (B, T_max) long.
    input_ids is padded with PAD on the right; attention_mask is 1 on real tokens and 0 on padding."""
    T = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    attention_mask = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        # TODO 1: copy s into the first len(s) slots of row i, and mark those slots with 1 in the mask
        pass  # ← your code here
    return input_ids, attention_mask


# ---- Given: the model -------------------------------------------------------
class Body(nn.Module):
    """A 2-block causal transformer. This is the part that gets pre-trained and then reused."""
    def __init__(self):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D, padding_idx=PAD), nn.Embedding(32, D)
        layer = nn.TransformerEncoderLayer(D, 4, 64, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, input_ids, attention_mask):
        T = input_ids.size(1)
        x = self.emb(input_ids) + self.pos(torch.arange(T))
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        return self.blocks(x, mask=causal, src_key_padding_mask=(attention_mask == 0))    # (B, T, D)


def lm_loss(body, lm_head, input_ids, attention_mask):
    """Next-token cross-entropy (CLM). Padded positions get label −100 so they are skipped."""
    logits = lm_head(body(input_ids, attention_mask)[:, :-1])
    labels = input_ids[:, 1:].clone()
    labels[attention_mask[:, 1:] == 0] = -100
    return F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)


def classify(body, head, input_ids, attention_mask):
    """Mean-pool the real tokens (the mask keeps padding out of the average), then the task head."""
    h = body(input_ids, attention_mask)
    m = attention_mask.unsqueeze(-1).float()
    return head((h * m).sum(1) / m.sum(1))                                                # (B, 2) logits


# ---------------------------------------------------------------------------
# TODO 2 — the learning-rate schedule
# ---------------------------------------------------------------------------
def lr_at(step, total_steps, warmup_steps, lr_max):
    """Linear warmup from 0 to lr_max over warmup_steps, then cosine decay to 0 at total_steps."""
    # TODO 2: two cases — step < warmup_steps, and the rest (math.cos, math.pi)
    return None  # ← your code here


# ---------------------------------------------------------------------------
# TODO 3 — one training step: the inside of trainer.train()
# ---------------------------------------------------------------------------
def train_step(body, head, optimizer, input_ids, attention_mask, labels, lr):
    for group in optimizer.param_groups:           # the scheduler's job: set this step's learning rate
        group["lr"] = lr
    params = list(body.parameters()) + list(head.parameters())
    # TODO 3: five lines, in this order —
    #   forward (classify) → loss (F.cross_entropy) → backward →
    #   clip (nn.utils.clip_grad_norm_(params, 1.0)) → optimizer.step() and optimizer.zero_grad()
    loss = None  # ← your code here
    return loss.item()


# ---- Given: one complete fine-tuning run ------------------------------------
def finetune(start_body, train_seqs, train_labels, lr, epochs=20, batch_size=8, warmup_ratio=0.06, seed=1):
    torch.manual_seed(seed)
    body, head = copy.deepcopy(start_body), nn.Linear(D, 2)        # keep the body, bolt on a fresh head
    optimizer = torch.optim.AdamW(list(body.parameters()) + list(head.parameters()), lr=lr, weight_decay=0.01)
    n = len(train_seqs)
    total = epochs * math.ceil(n / batch_size)
    warmup, step, t0 = max(1, int(warmup_ratio * total)), 0, time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            ids, mask = collate([train_seqs[j] for j in idx])
            train_step(body, head, optimizer, ids, mask, train_labels[idx], lr_at(step, total, warmup, lr))
            step += 1
    return body.eval(), head.eval(), time.perf_counter() - t0


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)

    # ---- TODO 1 --------------------------------------------------------------
    ids, mask = collate([[5, 6, 7, 8, 9], [3, 4, 5, 6], [1, 2, 3]])
    print("input_ids\n", ids, "\nattention_mask\n", mask)
    assert ids.shape == mask.shape == (3, 5) and mask.sum().item() == 12 and ids[2, 3:].tolist() == [PAD, PAD]
    print("✓ TODO 1: shape (B, T_max) = (3, 5); 12 real tokens, 3 pads (20% of the batch is padding)")

    # ---- TODO 2 --------------------------------------------------------------
    sched = [lr_at(s, 100, 10, 2e-5) for s in (0, 5, 10, 55, 100)]
    print("lr at steps 0, 5, 10, 55, 100:", [f"{x:.1e}" for x in sched])
    assert all(abs(a - b) < 1e-12 for a, b in zip(sched, [0.0, 1e-5, 2e-5, 1e-5, 0.0]))
    print("✓ TODO 2: matches the Part B hand calculation")

    # ---- Given: pre-training with CLM on UNLABELED sequences (≈ 10 s) -----------
    t0 = time.perf_counter()
    body, lm_head = Body(), nn.Linear(D, V)
    opt = torch.optim.AdamW(list(body.parameters()) + list(lm_head.parameters()), lr=3e-3)
    for _ in range(400):
        loss = lm_loss(body, lm_head, *collate(sample(0, 32, g) + sample(1, 32, g)))
        opt.zero_grad(); loss.backward(); opt.step()
    lm_val = collate(sample(0, 200, g) + sample(1, 200, g))
    with torch.no_grad():
        lm_before = lm_loss(body.eval(), lm_head, *lm_val).item()
    print(f"\npre-trained in {time.perf_counter() - t0:.0f} s: next-token loss {lm_before:.2f}  (a uniform guess scores ln 12 = {math.log(K):.2f})")

    # ---- The downstream task: 32 labeled examples ------------------------------
    train_seqs = sample(0, 16, g) + sample(1, 16, g)
    train_labels = torch.tensor([0] * 16 + [1] * 16)
    test_ids, test_mask = collate(sample(0, 300, g) + sample(1, 300, g))
    test_labels = torch.tensor([0] * 300 + [1] * 300)
    torch.manual_seed(5)
    scratch_body = Body()                                    # same architecture, random weights, never pre-trained

    def run(name, start_body, lr):
        b, h, secs = finetune(start_body, train_seqs, train_labels, lr)
        with torch.no_grad():
            acc = (classify(b, h, test_ids, test_mask).argmax(-1) == test_labels).float().mean().item()
            lm_after = lm_loss(b, lm_head, *lm_val).item()   # how much of the "schooling" survived?
        # TODO 4: return the experiments row for this run as a dict with the keys
        #         base_model, lr, batch_size, epochs, headline_metric, duration_seconds, lm_loss_after
        #         (batch_size is 8 and epochs is 20 here; round the metric to 3 places)
        return None  # ← your code here

    experiments = [run("from_scratch", scratch_body, 1e-3),
                   run("pretrained", body, 1e-4),            # too gentle
                   run("pretrained", body, 1e-3),            # the chisel
                   run("pretrained", body, 3e-2)]            # the sledgehammer
    print("\nleaderboard (sorted by headline_metric = test accuracy):")
    for row in sorted(experiments, key=lambda r: -r["headline_metric"]):
        print("  ", row)

    scratch, gentle, chisel, hammer = experiments
    assert chisel["headline_metric"] > scratch["headline_metric"] + 0.2, "pre-training should win clearly with 32 labels"
    assert chisel["headline_metric"] > gentle["headline_metric"] and chisel["headline_metric"] > hammer["headline_metric"]
    assert hammer["lm_loss_after"] > lm_before + 1.0, "a huge learning rate should wreck the pre-trained knowledge"
    print("\n✓ TODO 3 + 4: the loop trains; pre-trained beats from-scratch; the middle learning rate wins;")
    print(f"  the sledgehammer pushed next-token loss from {lm_before:.2f} to {hammer['lm_loss_after']:.2f} — catastrophic forgetting, measured.")
    # QUESTION (handout): this tiny model's sweet spot is lr = 1e-3. DistilBERT's is about 2e-5. Both runs follow the
    # same rule. State the rule in one sentence, then say which of the four rows you would deploy and why.
