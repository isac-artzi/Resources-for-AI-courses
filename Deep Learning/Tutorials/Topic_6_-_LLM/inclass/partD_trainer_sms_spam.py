"""
AIT-204 · Topic 6 · Part D — The real thing: three Trainer runs on SMS spam (at home, ~30–60 min on a laptop CPU)

    pip install "transformers>=4.41" datasets torch accelerate

This is FineTuneFleet build steps 3–6 in one file: load and split the corpus, record a baseline,
fine-tune distilbert-base-uncased three times, and print one `experiments` row per run with a
95% bootstrap confidence interval on F1. Nothing here talks to Supabase; the rows are printed so
you can see exactly what your db layer has to INSERT.

First run downloads about 270 MB (model + dataset). Set SUBSET = None for the full corpus once it works.
"""
import time

import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

BASE = "distilbert-base-uncased"
SUBSET = 1500            # messages used in total; None = all 5,574. Keep it small for a first CPU run.
SEED = 204


# ---- metrics (your Exercise 3 code) -----------------------------------------
def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1)); fp = np.sum((y_true == 0) & (y_pred == 1)); fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=0):
    rng, n = np.random.default_rng(seed), len(y_true)
    scores = [f1_score(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"f1": f1_score(labels, np.argmax(logits, axis=-1))}


# ---- build step 3: dataset + a documented split ------------------------------
ds = load_dataset("ucirvine/sms_spam", split="train")              # columns: "sms" (text), "label" (0 = ham, 1 = spam)
ds = ds.shuffle(seed=SEED)
if SUBSET:
    ds = ds.select(range(SUBSET))
split = ds.train_test_split(test_size=0.2, seed=SEED, stratify_by_column="label")
held = split["test"].train_test_split(test_size=0.5, seed=SEED, stratify_by_column="label")
train, val, test = split["train"], held["train"], held["test"]     # 80 / 10 / 10
print(f"train {len(train)}  validation {len(val)}  test {len(test)}  spam share {np.mean(train['label']):.3f}")

tok = AutoTokenizer.from_pretrained(BASE)
encode = lambda batch: tok(batch["sms"], truncation=True, max_length=128)     # pad per batch, not to 128
train, val, test = (d.map(encode, batched=True) for d in (train, val, test))
y_test = np.array(test["label"])

# ---- build step 4: a baseline row --------------------------------------------
# The untrained head on top of the pre-trained body: what you get with zero task-specific training.
# (For your product's zero_shot row, use the off-the-shelf classifier your team chose in step 4.)
set_seed(SEED)
baseline = Trainer(model=AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2),
                   args=TrainingArguments(output_dir="out/baseline", per_device_eval_batch_size=64, report_to="none"),
                   data_collator=DataCollatorWithPadding(tok))
pred = np.argmax(baseline.predict(test).predictions, axis=-1)
rows = [dict(base_model="untrained_head", lr=None, batch_size=None, epochs=0, headline_metric=round(f1_score(y_test, pred), 3),
             ci=tuple(round(x, 3) for x in bootstrap_ci(y_test, pred)), duration_seconds=0.0)]

# ---- build step 5: three configurations, one row each ---------------------------
for lr, batch_size, epochs in [(2e-5, 16, 2), (5e-5, 16, 2), (5e-5, 32, 3)]:
    set_seed(SEED)                                                 # seeded → reproducible
    args = TrainingArguments(
        output_dir=f"out/lr{lr}_bs{batch_size}_ep{epochs}",
        num_train_epochs=epochs, per_device_train_batch_size=batch_size, per_device_eval_batch_size=64,
        learning_rate=lr, warmup_ratio=0.06, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=20, report_to="none", seed=SEED)
    trainer = Trainer(model=AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2), args=args,
                      train_dataset=train, eval_dataset=val,      # choose the checkpoint on VALIDATION …
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    t0 = time.perf_counter()
    trainer.train()
    secs = time.perf_counter() - t0
    pred = np.argmax(trainer.predict(test).predictions, axis=-1)   # … and report on TEST, once
    rows.append(dict(base_model=BASE, lr=lr, batch_size=batch_size, epochs=epochs, headline_metric=round(f1_score(y_test, pred), 3),
                     ci=tuple(round(x, 3) for x in bootstrap_ci(y_test, pred)), duration_seconds=round(secs, 1)))

# ---- build step 6: the leaderboard --------------------------------------------
print("\nexperiments (sorted by test F1):")
for row in sorted(rows, key=lambda r: -r["headline_metric"]):
    print("  ", row)
# POST TO THE PADLET: this table, your machine (chip, RAM), and one sentence answering —
# do the confidence intervals of your best two runs overlap, and what does that mean for which one you deploy?
