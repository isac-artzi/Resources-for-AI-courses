"""
AIT-204 · Topic 6 · In-class Exercise 4 — Fine-tune or prompt? LoRA, the cost story and the lab notebook
Time: ~15–20 min (+ Part D at home)   |   Needs: torch (sqlite3 and hashlib ship with Python)

    LoRA        : freeze W, learn a low-rank correction   W' = W + (alpha / r) · B A
    cost        : pay per token to a frontier API, or a flat rate for your own small model
    persistence : every run is a row in `experiments`; every prediction is a row in `predictions`
                  that stores a HASH of the request, never the text

sqlite3 stands in for Supabase Postgres here. The tables match the FineTuneFleet migration.
Fill in every "TODO".
"""
import hashlib
import sqlite3

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# TODO 1 + 2 — a LoRA layer
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        for p in self.base.parameters():
            p.requires_grad = False                                  # the textbook stays as printed
        d_out, d_in = base.weight.shape
        # TODO 1a: the two sticky notes. A is (r, d_in) with small random values (std 0.01);
        #          B is (d_out, r) and starts at ZERO, so the layer begins identical to `base`.
        self.A = None  # ← your code here
        self.B = None  # ← your code here

    def forward(self, x):
        # TODO 1b: base(x) plus the low-rank path  scale · x Aᵀ Bᵀ
        return None  # ← your code here

    def merge(self) -> nn.Linear:
        """Glue the notes into the page: return a plain nn.Linear whose weight is W + scale · B A."""
        merged = nn.Linear(self.base.in_features, self.base.out_features)
        with torch.no_grad():
            # TODO 2: set merged.weight and merged.bias (use .copy_)
            pass  # ← your code here
        return merged


def n_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# TODO 3 — the cost story
# ---------------------------------------------------------------------------
def api_monthly_cost(requests, tokens_per_request, usd_per_million_tokens):
    # TODO 3a: one line
    return None  # ← your code here


def break_even_requests(hosting_usd_per_month, tokens_per_request, usd_per_million_tokens):
    """Monthly volume at which a flat-rate small model costs the same as the per-token API."""
    # TODO 3b: one line
    return None  # ← your code here


# ---------------------------------------------------------------------------
# TODO 4 — the lab notebook: experiments + predictions
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE experiments (id INTEGER PRIMARY KEY, base_model TEXT, lr REAL, batch_size INTEGER, epochs INTEGER,
                          headline_metric REAL, duration_seconds REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE predictions (request_hash TEXT, fine_tuned_output TEXT, zero_shot_output TEXT, latency_ms REAL,
                          served_by_experiment_id INTEGER REFERENCES experiments(id),
                          created_at TEXT DEFAULT CURRENT_TIMESTAMP);
"""


def request_hash(text: str) -> str:
    # TODO 4a: SHA-256 of the UTF-8 bytes, as a hex string
    return None  # ← your code here


def log_prediction(conn, text, fine_tuned_output, zero_shot_output, latency_ms, experiment_id):
    # TODO 4b: INSERT one row into predictions. Use ? placeholders. The raw text must not be stored.
    pass  # ← your code here


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- TODO 1: LoRA starts as a no-op and trains ~2% of the layer ------------
    base = nn.Linear(768, 768)
    lora = LoRALinear(base, r=8, alpha=16)
    x = torch.randn(16, 768)
    assert torch.allclose(lora(x), base(x)), "with B = 0 the LoRA layer must equal the base layer"
    full = base.weight.numel()
    print(f"full fine-tune: {full:,} weights    LoRA r=8: {n_trainable(lora):,} trainable    ratio {full / n_trainable(lora):.0f}×")
    assert n_trainable(lora) == 2 * 768 * 8 == 12_288
    print("✓ TODO 1: identical to the base at step 0; only A and B train")

    # ---- TODO 2: train the notes a little, then merge ---------------------------
    opt = torch.optim.SGD([lora.A, lora.B], lr=0.1)
    target = torch.randn(16, 768)
    w_before = base.weight.clone()
    for _ in range(20):
        loss = ((lora(x) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert torch.equal(base.weight, w_before), "the frozen base must not move"
    merged = lora.merge()
    assert torch.allclose(merged(x), lora(x), atol=1e-5)
    delta = merged.weight - base.weight
    print(f"rank of the learned update ΔW = {torch.linalg.matrix_rank(delta).item()}  (never more than r = {lora.r})")
    assert torch.linalg.matrix_rank(delta).item() <= lora.r
    print("✓ TODO 2: merged layer == LoRA layer, base untouched, ΔW has rank ≤ r")

    # ---- TODO 3: the Part B numbers ---------------------------------------------
    cost = api_monthly_cost(100_000, 300, 3.00)
    be = break_even_requests(25.00, 300, 3.00)
    print(f"API: ${cost:,.2f} / month at 100,000 requests    break-even vs a $25 host: {be:,.0f} requests / month")
    assert abs(cost - 90.0) < 1e-9 and round(be) == 27_778
    print("✓ TODO 3: matches the hand calculation (the prices are assumptions — state them next to the number)")

    # ---- TODO 4: every run and every prediction is a row --------------------------
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    runs = [("zero_shot", None, None, None, 0.777, 0.0),
            ("distilbert-base-uncased", 2e-5, 16, 3, 0.840, 412.0),
            ("distilbert-base-uncased", 5e-5, 16, 3, 0.861, 409.0),
            ("distilbert-base-uncased", 5e-5, 32, 5, 0.852, 655.0)]
    conn.executemany("INSERT INTO experiments (base_model, lr, batch_size, epochs, headline_metric, duration_seconds) "
                     "VALUES (?, ?, ?, ?, ?, ?)", runs)
    best_id, best_f1 = conn.execute("SELECT id, headline_metric FROM experiments ORDER BY headline_metric DESC LIMIT 1").fetchone()
    sms = "WINNER!! Claim your free prize now"
    log_prediction(conn, sms, "spam", "ham", 41.7, best_id)
    row = conn.execute("SELECT * FROM predictions").fetchone()
    print("predictions row →", row[:5])
    assert row[0] == request_hash(sms) and len(row[0]) == 64
    assert all(sms not in str(col) for col in row), "the raw SMS must never be stored"
    assert request_hash(sms) != request_hash(sms + "!"), "one extra character → a different fingerprint"
    lr, f1 = conn.execute("SELECT e.lr, e.headline_metric FROM predictions p "
                          "JOIN experiments e ON e.id = p.served_by_experiment_id").fetchone()
    print(f"audit: this prediction was served by experiment {best_id} (lr = {lr}, F1 = {f1})")
    assert f1 == best_f1 == 0.861
    print("✓ TODO 4: hash stored, text not stored, prediction joins to the experiment that served it")

    print("\nleaderboard (what GET /experiments returns):")
    for r_ in conn.execute("SELECT base_model, lr, batch_size, epochs, headline_metric FROM experiments ORDER BY headline_metric DESC"):
        print("  ", r_)
    # QUESTION (handout, feeds DQ 2–4 and the Cost & Latency tab): the customer handles 20,000 SMS a month.
    # Using your break-even number, which option do you recommend, and what besides cost belongs in the answer?
