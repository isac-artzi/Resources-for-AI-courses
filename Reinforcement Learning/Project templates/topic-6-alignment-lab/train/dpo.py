"""
train/dpo.py — direct preference optimisation.

TRAINING TIER (imports torch). Never imported by api/ or ui/.

    python -m train.dpo --offline --betas 0.05 0.1 0.5

THE LOSS IS THE POINT. Everything else in this file is scaffolding so that the
loss has something to run on. `dpo_loss` below is seven lines of body; build
step 3(b) of the discussion questions asks for it in fewer than fifteen, and
the reason it can be that short is the derivation, not the code.

WHERE THE LOSS COMES FROM (the three-line version of DQ 3(a))
------------------------------------------------------------
The KL-regularised objective  max_pi  E[r(x,y)] - beta*KL(pi || pi_ref)  has
the closed-form optimum

    pi*(y|x) = (1 / Z(x)) * pi_ref(y|x) * exp( r(x,y) / beta )

Solve that for the reward:

    r(x,y) = beta * log( pi*(y|x) / pi_ref(y|x) ) + beta * log Z(x)

Substitute into the Bradley-Terry loss from `train/reward_model.py`. The loss
depends on r only through the DIFFERENCE r(x, y_c) - r(x, y_r), and both terms
carry the same beta*log Z(x) because Z depends on the prompt alone — so it
cancels. The intractable partition function disappears, and what is left is a
loss you can evaluate with two forward passes:

    L = - log sigma( beta * [ (log pi(y_c|x) - log pi_ref(y_c|x))
                            - (log pi(y_r|x) - log pi_ref(y_r|x)) ] )

The reward model has not been approximated away. It has been REPARAMETERISED:
the policy is its own reward model, and `beta * log(pi/pi_ref)` is called the
IMPLICIT REWARD for that reason.

WHY THIS PRODUCT BUILDS ON DPO RATHER THAN REPRODUCING INSTRUCTGPT
-----------------------------------------------------------------
A practical reason, not an aesthetic one. In TRL 1.9.x, `DPOTrainer`,
`RewardTrainer`, `SFTTrainer` and `GRPOTrainer` are stable, while `PPOTrainer`
is marked EXPERIMENTAL. A PPO-based RLHF pipeline also needs four models
resident at once — policy, reference, reward model, and value head — against
DPO's two, which is the difference between fitting on a free Colab T4 and not.
Argue in your README about which you would choose given a fixed compute budget;
this template ships the one you can actually run.

THE OFFLINE FALLBACK
--------------------
`TinyLM` is a self-contained trigram-context language model, a few thousand
parameters, trained here from scratch. It exists so this file runs end to end
in a sandbox with no model hub. It is a real autoregressive language model —
it has a vocabulary, a softmax over next tokens, and a sequence log-probability
— which is all the DPO loss requires. It is not gpt2 and does not pretend to be.

Two things the tiny model buys that gpt2 does not, and which are worth keeping
even after you switch to the real path:

  * **KL FROM THE REFERENCE IS EXACT.** The vocabulary is a few hundred types,
    so the full next-token distribution is computable at every position and the
    per-token KL is a sum, not a sample estimate. On gpt2 you estimate it and
    the estimator's variance is comparable to the effect you are measuring at
    large beta.
  * **THE DEGENERATE SOLUTION IS VISIBLE.** At small beta the policy collapses
    onto repeated high-reward tokens within a few hundred steps. That collapse
    is exactly what `train/reward_hacking.py` measures, and on gpt2 it takes
    long enough that most students never see it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

REPORTS = pathlib.Path("reports")

BOS = "<bos>"
EOS = "<eos>"


def pin_threads(n: int = 1) -> None:
    """Force PyTorch to one intra-op thread. Call before any tensor work here.

    This looks like a pessimisation and is a large speed-up for THIS model, and
    the reason generalises. The tiny LM's largest matrix multiply is 96x64;
    dispatching that across eight worker threads costs more in synchronisation
    than the arithmetic costs in total, and on a shared or containerised host
    the workers also contend with each other. Measured on the offline sweep:
    ~95 s pinned to one thread against >6 minutes with the default pool.

    Do NOT copy this into a run with a real transformer, where the matrices are
    large enough for the pool to pay for itself. It is a decision about model
    size, not a style rule.
    """
    import torch

    torch.set_num_threads(max(1, n))


# ===========================================================================
# THE LOSS — build step 3(b): fewer than fifteen lines.
# ===========================================================================


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta: float = 0.1,
):
    """The DPO objective. Inputs are SEQUENCE log-probabilities, one per pair.

    Returns `(loss, chosen_implicit_reward, rejected_implicit_reward)`, matching
    what `trl.DPOTrainer` reports so your numbers are comparable with anyone
    else's.

    `logsigmoid` rather than `log(sigmoid(x))` for the same reason as in
    `train/reward_model.py`: the second underflows to -inf and NaNs the batch.

    The rewards are DETACHED. They are diagnostics, and a diagnostic that
    carries a gradient is a second, unintended term in your objective.
    """
    import torch.nn.functional as F

    chosen_logratio = policy_chosen_logps - reference_chosen_logps
    rejected_logratio = policy_rejected_logps - reference_rejected_logps
    logits = chosen_logratio - rejected_logratio          # the implicit-reward margin / beta
    loss = -F.logsigmoid(beta * logits).mean()
    return loss, (beta * chosen_logratio).detach(), (beta * rejected_logratio).detach()


def verify_loss_decreases_with_margin(betas=(0.05, 0.1, 0.5)) -> dict:
    """Numerical check that the loss falls as the chosen-response margin grows.

    Required by DQ 3(b) and worth having as a function rather than as a
    paragraph: it is the cheapest possible test that the sign convention is
    right. A DPO implementation with the two log-ratios swapped trains happily,
    reports a falling loss, and produces a policy that has learned to prefer the
    REJECTED response — and nothing else in the pipeline notices.
    """
    import torch

    out: dict[str, list[float]] = {}
    margins = np.linspace(-2.0, 4.0, 13)
    for beta in betas:
        losses = []
        for m in margins:
            # A synthetic pair: the reference is neutral, and the policy assigns
            # the chosen response `m` more log-probability than the rejected one.
            pc = torch.tensor([float(m)])
            pr = torch.tensor([0.0])
            rc = torch.tensor([0.0])
            rr = torch.tensor([0.0])
            loss, _, _ = dpo_loss(pc, pr, rc, rr, beta=beta)
            losses.append(float(loss))
        out[f"beta={beta}"] = losses
    out["margins"] = margins.tolist()
    out["monotone_decreasing"] = all(
        all(np.diff(v) < 1e-9) for k, v in out.items() if k.startswith("beta=")
    )
    return out


# ===========================================================================
# The tiny language model
# ===========================================================================


@dataclass
class LMConfig:
    embed_dim: int = 32
    hidden: int = 64
    context: int = 2          # trigram: the model conditions on the last two tokens
    max_new_tokens: int = 28
    seed: int = 0


class Vocab:
    """Closed vocabulary with BOS and EOS. Small enough that KL is exact."""

    def __init__(self, tokens: list[str]) -> None:
        self.itos = [BOS, EOS] + [t for t in tokens if t not in (BOS, EOS)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        from shared.preprocess import tokenise

        # Unknown tokens are DROPPED, not mapped to an <unk> id. With a closed
        # vocabulary built from the same generator that produced the text there
        # should be none; if there are, silently learning a distribution over
        # <unk> is worse than losing the token, because the model would then
        # generate <unk> and the reward head would score the literal string.
        return [self.stoi[t] for t in tokenise(text) if t in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] for i in ids if i not in (self.stoi[BOS], self.stoi[EOS]))


def build_lm(vocab_size: int, cfg: LMConfig):
    """A context-window language model: p(y_t | y_{t-2}, y_{t-1}, prompt).

    Written as an explicit module rather than as a Sequential because the
    forward pass has to concatenate a prompt representation with the context
    embeddings, and hiding that in a lambda would hide the one interesting
    thing about the architecture.
    """
    import torch
    from torch import nn

    torch.manual_seed(cfg.seed)

    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = nn.Embedding(vocab_size, cfg.embed_dim)
            self.net = nn.Sequential(
                nn.Linear(cfg.embed_dim * (cfg.context + 1), cfg.hidden),
                nn.ReLU(),
                nn.Linear(cfg.hidden, vocab_size),
            )
            self.context = cfg.context

        def prompt_vector(self, prompt_ids: "torch.Tensor") -> "torch.Tensor":
            """Bag-of-embeddings over the prompt. (B, T_p) -> (B, E).

            Order-insensitive on purpose: this is a conditioning signal, not a
            second language model, and giving it sequence structure would double
            the parameter count for a component the experiment is not about.
            """
            mask = (prompt_ids >= 0).float().unsqueeze(-1)
            emb = self.tok(prompt_ids.clamp(min=0)) * mask
            return emb.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        def forward(self, prompt_ids, context_ids):
            """(B, T_p), (B, C) -> (B, V) logits for the next token."""
            pv = self.prompt_vector(prompt_ids)
            ctx = self.tok(context_ids.clamp(min=0)).reshape(context_ids.shape[0], -1)
            return self.net(torch.cat([pv, ctx], dim=-1))

    return TinyLM()


def _contexts(seq_ids: list[int], bos: int, context: int) -> list[list[int]]:
    """The rolling context window for each position of a sequence.

    Padded at the front with BOS rather than with a zero id. Zero is a real
    token id in this vocabulary, so front-padding with it would teach the model
    that every sequence starts with two copies of `<bos>`'s neighbour.
    """
    padded = [bos] * context + seq_ids
    return [padded[i : i + context] for i in range(len(seq_ids))]


def sequence_logprob(model, vocab: Vocab, prompt_ids, sequences, device="cpu"):
    """Sum of log p(y_t | context, prompt) over a batch of sequences.

    Returns a (B,) tensor.

    THE EOS TOKEN IS INCLUDED IN THE SUM. Omitting it is a real and popular bug:
    the model then pays nothing for never stopping, and DPO — which rewards
    raising the log-probability of the chosen sequence — will happily learn a
    policy that generates until the length cap on every prompt.

    The sequences are ragged, so instead of padding to a rectangle they are
    FLATTENED into one long batch of (context, target) rows and reassembled
    with `index_add`. Two reasons this is worth the extra six lines:

      * A padded rectangle at these length distributions is about 40% padding,
        and the masking needed to keep the padding out of the sum is exactly
        the kind of thing that silently stops working when a sequence is
        longer than the pad width.
      * It makes the whole batch ONE forward pass. The obvious per-sequence
        loop is a hundred times slower here, and this function is called four
        times per DPO step — it is the inner loop of the whole file.

    The window construction is done in NumPy with fancy indexing rather than in
    a Python loop for the same reason: at 300 SFT steps x 64 sequences x 20
    tokens, the Python-level loop version spends more time building index lists
    than PyTorch spends on arithmetic.
    """
    import torch

    eos = vocab.stoi[EOS]
    bos = vocab.stoi[BOS]
    ctx_n = model.context

    lengths = [len(s) + 1 for s in sequences]      # +1 for the EOS target
    total = sum(lengths)
    ctx = np.empty((total, ctx_n), dtype=np.int64)
    tgt = np.empty(total, dtype=np.int64)
    owner = np.empty(total, dtype=np.int64)

    window = np.arange(ctx_n)[None, :]
    pos = 0
    for b, seq in enumerate(sequences):
        full = np.asarray(list(seq) + [eos], dtype=np.int64)
        n = full.shape[0]
        padded = np.concatenate([np.full(ctx_n, bos, dtype=np.int64), full])
        ctx[pos : pos + n] = padded[np.arange(n)[:, None] + window]
        tgt[pos : pos + n] = full
        owner[pos : pos + n] = b
        pos += n

    owner_t = torch.from_numpy(owner).to(device)
    ctx_t = torch.from_numpy(ctx).to(device)
    tgt_t = torch.from_numpy(tgt).to(device)
    # Advanced indexing rather than a stack of per-row tensors: the prompt for
    # every flattened row is `prompt_ids[owner]`, one gather instead of `total`
    # Python-level tensor constructions.
    pr_t = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)[owner_t]

    logits = model(pr_t, ctx_t)
    logps = torch.log_softmax(logits, dim=-1).gather(1, tgt_t.unsqueeze(1)).squeeze(1)

    out = torch.zeros(len(sequences), device=device)
    return out.index_add(0, owner_t, logps)


def pad_prompts(vocab: Vocab, prompts: list[str], width: int | None = None):
    """Encode prompts to a rectangular (B, T) tensor, padded with -1.

    -1 rather than 0, because `prompt_vector` masks on `>= 0` and 0 is the id
    of `<bos>`. Padding with a real token id would average that token into
    every short prompt's representation.
    """
    import torch

    enc = [vocab.encode(p) for p in prompts]
    width = width or max((len(e) for e in enc), default=1)
    rows = [e[:width] + [-1] * (width - len(e[:width])) for e in enc]
    return torch.tensor(rows, dtype=torch.long)


# ===========================================================================
# Stage 1: supervised fine-tuning, which produces the REFERENCE policy
# ===========================================================================


def sft(model, vocab: Vocab, rows: list[dict], steps: int = 400, lr: float = 3e-3,
        batch_size: int = 64, seed: int = 0, log_every: int = 0) -> list[dict]:
    """Maximum likelihood on the CHOSEN responses. Stage 1 of the three-stage pipeline.

    This is where the reference policy comes from, and the reference policy is
    not an implementation detail: it defines what `KL from reference` is a
    distance from, so every KL number in this product is relative to a model
    fitted right here. Two students with different SFT budgets will report
    different KLs for the same beta and both will be correct. Say what your SFT
    budget was.
    """
    import torch

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        idx = rng.integers(0, len(rows), size=min(batch_size, len(rows)))
        batch = [rows[int(i)] for i in idx]
        prompts = pad_prompts(vocab, [r["prompt"] for r in batch])
        seqs = [vocab.encode(r["chosen"]) for r in batch]
        # A pair whose chosen response encodes to nothing would contribute an
        # empty sequence and a zero-length index_add — harmless, but it silently
        # shrinks the effective batch. Drop it visibly instead.
        keep = [i for i, s in enumerate(seqs) if s]
        if not keep:
            continue
        logp = sequence_logprob(model, vocab, prompts[keep], [seqs[i] for i in keep])
        loss = -logp.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and step % log_every == 0:
            history.append({"step": step, "nll": float(loss.detach())})
    return history


# ===========================================================================
# Stage 3: DPO
# ===========================================================================


@dataclass
class DPOResult:
    beta: float
    final_loss: float
    implicit_reward_margin: float
    implicit_reward_accuracy: float
    kl_from_reference: float
    steps: int
    seed: int
    history: list[dict] = field(default_factory=list)


def train_dpo(
    policy,
    reference,
    vocab: Vocab,
    train_rows: list[dict],
    test_rows: list[dict],
    beta: float,
    steps: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> DPOResult:
    """Run DPO at one beta. `policy` is trained in place; `reference` is frozen.

    The reference must be a SEPARATE FROZEN COPY, not the same object and not
    a copy that shares parameters. `copy.deepcopy` plus `requires_grad_(False)`
    is the whole of it; getting it wrong produces a log-ratio that is
    identically zero, a loss pinned at -log sigma(0) = 0.693, and a training
    curve that is a flat line nobody questions because 0.693 looks like a
    plausible loss.
    """
    import torch

    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history: list[dict] = []

    for step in range(steps):
        idx = rng.integers(0, len(train_rows), size=min(batch_size, len(train_rows)))
        batch = [train_rows[int(i)] for i in idx]
        prompts = pad_prompts(vocab, [r["prompt"] for r in batch])
        chosen = [vocab.encode(r["chosen"]) for r in batch]
        rejected = [vocab.encode(r["rejected"]) for r in batch]
        keep = [i for i in range(len(batch)) if chosen[i] and rejected[i]]
        if not keep:
            continue
        p = prompts[keep]
        yc = [chosen[i] for i in keep]
        yr = [rejected[i] for i in keep]

        pc = sequence_logprob(policy, vocab, p, yc)
        pr = sequence_logprob(policy, vocab, p, yr)
        with torch.no_grad():
            rc = sequence_logprob(reference, vocab, p, yc)
            rr = sequence_logprob(reference, vocab, p, yr)

        loss, imp_c, imp_r = dpo_loss(pc, pr, rc, rr, beta=beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 25 == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "margin": float((imp_c - imp_r).mean()),
                    "accuracy": float((imp_c > imp_r).float().mean()),
                }
            )

    evaluated = evaluate_dpo(policy, reference, vocab, test_rows, beta)
    return DPOResult(
        beta=beta,
        final_loss=history[-1]["loss"] if history else float("nan"),
        steps=steps,
        seed=seed,
        history=history,
        **evaluated,
    )


def evaluate_dpo(policy, reference, vocab: Vocab, rows: list[dict], beta: float) -> dict:
    """Implicit reward margin, implicit reward accuracy, and KL — on HELD-OUT pairs.

    Evaluating these on the training pairs is the mistake that makes DPO look
    like it worked at every beta: the implicit reward accuracy on data the
    policy was fitted to goes to 1.0 whatever the KL constraint was, because
    that is what the objective maximises.
    """
    import torch

    with torch.no_grad():
        prompts = pad_prompts(vocab, [r["prompt"] for r in rows])
        chosen = [vocab.encode(r["chosen"]) for r in rows]
        rejected = [vocab.encode(r["rejected"]) for r in rows]
        keep = [i for i in range(len(rows)) if chosen[i] and rejected[i]]
        p = prompts[keep]
        yc = [chosen[i] for i in keep]
        yr = [rejected[i] for i in keep]

        pc = sequence_logprob(policy, vocab, p, yc)
        pr = sequence_logprob(policy, vocab, p, yr)
        rc = sequence_logprob(reference, vocab, p, yc)
        rr = sequence_logprob(reference, vocab, p, yr)

        imp_c = beta * (pc - rc)
        imp_r = beta * (pr - rr)
        kl = exact_kl(policy, reference, vocab, [r["prompt"] for r in rows][:64])

    return {
        "implicit_reward_margin": float((imp_c - imp_r).mean()),
        "implicit_reward_accuracy": float((imp_c > imp_r).float().mean()),
        "kl_from_reference": kl,
    }


def exact_kl(policy, reference, vocab: Vocab, prompts: list[str],
             max_new_tokens: int = 28, seed: int = 0) -> float:
    """Mean over prompts of sum_t KL( pi(.|s_t) || pi_ref(.|s_t) ), in nats.

    Computed EXACTLY at each position, not estimated from samples. The
    trajectory is sampled from the POLICY — which is the correct measure for
    the KL term in the RLHF objective, since that expectation is under pi —
    but given a prefix the per-token KL is a closed-form sum over the
    vocabulary, and this vocabulary is small enough to sum over.

    On a real language model you cannot do this and you use the k1 or k3
    estimator instead. Note in your README which you used: the sampled
    estimator has variance comparable to the beta effect at large beta, and a
    KL curve that looks noisy at beta = 0.5 is usually the estimator rather
    than the run.
    """
    import torch

    total = 0.0
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            ids = generate(policy, vocab, [prompt], max_new_tokens=max_new_tokens,
                           seed=seed + i, return_ids=True)[0]
            if not ids:
                continue
            p_ids = pad_prompts(vocab, [prompt])
            ctxs = _contexts(ids, vocab.stoi[BOS], policy.context)
            ctx_t = torch.tensor(ctxs, dtype=torch.long)
            pr_t = p_ids.repeat(len(ctxs), 1)
            lp = torch.log_softmax(policy(pr_t, ctx_t), dim=-1)
            lq = torch.log_softmax(reference(pr_t, ctx_t), dim=-1)
            # sum_v p(v) * (log p(v) - log q(v)), summed over positions.
            total += float((lp.exp() * (lp - lq)).sum())
    return total / max(len(prompts), 1)


def generate(model, vocab: Vocab, prompts: list[str], max_new_tokens: int = 28,
             temperature: float = 1.0, seed: int = 0, return_ids: bool = False):
    """Sample completions. Offline, in the training tier, exactly once.

    The service never calls this. Its output is written to `completions` and
    `GET /completions` reads the rows back — see the architecture note.

    Sampling rather than greedy decoding, deliberately: a greedy decode from a
    degenerate policy and a greedy decode from a healthy one can look equally
    repetitive, because greedy decoding of ANY language model repeats. The
    reward-hacking result would then be an artefact of the decoder rather than
    of the alignment run.
    """
    import torch

    eos = vocab.stoi[EOS]
    bos = vocab.stoi[BOS]
    g = torch.Generator().manual_seed(seed)
    out = []
    with torch.no_grad():
        for prompt in prompts:
            p_ids = pad_prompts(vocab, [prompt])
            ids: list[int] = []
            for _ in range(max_new_tokens):
                ctx = torch.tensor([_contexts(ids + [0], bos, model.context)[-1]],
                                   dtype=torch.long)
                logits = model(p_ids, ctx)[0] / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1, generator=g))
                if nxt == eos:
                    break
                if nxt == bos:
                    continue   # never emit BOS mid-sequence; it is not a word
                ids.append(nxt)
            out.append(ids if return_ids else vocab.decode(ids))
    return out


# ===========================================================================
# The real path
# ===========================================================================


def train_dpo_trl(dataset, beta: float, model_name: str = "gpt2",
                  output_dir: str = "runs/dpo", steps: int = 500, seed: int = 0):
    """The required path: TRL's DPOTrainer on a small causal LM.

    Imported inside the function because `trl` and `transformers` are
    training-tier dependencies and this module must stay importable without
    them.

    Version note that will save you an afternoon: TRL is at 1.9.x, where
    `DPOTrainer` takes `DPOConfig` (not `TrainingArguments`) and expects a
    dataset with the columns `prompt`, `chosen`, `rejected` as PLAIN STRINGS.
    `DPOTrainer` also builds the reference model itself if you pass
    `ref_model=None`, which is what you want — a hand-rolled reference that
    shares parameters with the policy is the silent-flat-loss bug described in
    `train_dpo` above.

    `beta` is the same beta as everywhere else in this file, and TRL logs
    `rewards/margins`, `rewards/accuracies` and `logps/*` under names that map
    directly onto the `alignment_runs` columns. Log them to Supabase rather
    than reading them off a TensorBoard screenshot.
    """
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        # gpt2 ships no pad token. Setting it to EOS is the standard fix; the
        # attention mask is what actually stops the padding contributing, so
        # this is a bookkeeping choice rather than a modelling one.
        tok.pad_token = tok.eos_token

    ds = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
         for r in dataset.train]
    )
    cfg = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        max_steps=steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,     # 5e-6, not 5e-5. DPO on a pretrained model is
                                # fragile; a policy that moves too fast leaves
                                # the reference's support and the log-ratios
                                # become meaningless before the loss looks bad.
        logging_steps=10,
        seed=seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=AutoModelForCausalLM.from_pretrained(model_name),
        ref_model=None,          # TRL makes the frozen copy for you
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()
    return trainer


# ===========================================================================
# The end-to-end alignment run
# ===========================================================================


def run_alignment(
    dataset,
    betas: tuple[float, ...] = (0.05, 0.1, 0.5),
    sft_steps: int = 400,
    dpo_steps: int = 300,
    n_gen_prompts: int = 60,
    seed: int = 0,
    store=None,
    experiment_id: str | None = None,
    score_fn=None,
    reports: pathlib.Path = REPORTS,
    notes: str = "tiny-lm offline fallback",
) -> dict:
    """SFT once, then DPO at each beta; generate, score, and persist everything.

    Order matters and is not arbitrary:

      1. SFT on the chosen responses -> the REFERENCE policy. Done once and
         shared by every beta, so the KL numbers are distances from the same
         point. Re-running SFT per beta would make them incomparable.
      2. DPO from a fresh copy of the reference at each beta. Fresh, not
         continued from the previous beta — otherwise the beta = 0.5 run starts
         from a policy that beta = 0.05 has already wrecked.
      3. Generate from the reference (the 'base' variant) and from each aligned
         policy, for the same held-out prompts.
      4. Score every completion with the DEPLOYED reward head, through the real
         `POST /score` handler, so the numbers in the report are the numbers the
         service returns.
    """
    import copy

    import torch

    from train.data import SyntheticCorpus, true_quality

    if store is None:
        from shared.store import get_store

        store = get_store()

    pin_threads()
    torch.manual_seed(seed)
    corpus = SyntheticCorpus(seed=seed)
    vocab = Vocab(corpus.vocabulary())
    cfg = LMConfig(seed=seed)

    reference = build_lm(len(vocab), cfg)
    sft_hist = sft(reference, vocab, dataset.train, steps=sft_steps, seed=seed, log_every=50)
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()

    # Held-out prompts, deduplicated. Build step 6 requires at least 50.
    gen_prompts = list(dict.fromkeys(r["prompt"] for r in dataset.test))[:n_gen_prompts]
    prompt_ids = {r["prompt"]: r["prompt_id"] for r in dataset.test}

    if score_fn is None:
        score_fn = _deployed_score_fn()

    results: list[dict] = []
    completions: list[dict] = []

    # -- the base model: beta is None, because it IS the reference -----------
    base_texts = generate(reference, vocab, gen_prompts, cfg.max_new_tokens, seed=seed)
    completions += _completion_rows(
        gen_prompts, prompt_ids, base_texts, "base", None, score_fn, true_quality, experiment_id
    )

    for beta in betas:
        policy = copy.deepcopy(reference)
        for p in policy.parameters():
            p.requires_grad_(True)
        policy.train()
        res = train_dpo(policy, reference, vocab, dataset.train, dataset.test,
                        beta=beta, steps=dpo_steps, seed=seed)
        policy.eval()

        texts = generate(policy, vocab, gen_prompts, cfg.max_new_tokens, seed=seed)
        rows = _completion_rows(
            gen_prompts, prompt_ids, texts, "dpo", beta, score_fn, true_quality, experiment_id
        )
        completions += rows

        scores = [r["reward_score"] for r in rows if r["reward_score"] is not None]
        quals = [r["true_quality"] for r in rows if r["true_quality"] is not None]
        run_row = {
            "experiment_id": experiment_id,
            "beta": beta,
            "final_loss": res.final_loss,
            "implicit_reward_margin": res.implicit_reward_margin,
            "implicit_reward_accuracy": res.implicit_reward_accuracy,
            "kl_from_reference": res.kl_from_reference,
            "mean_reward_model_score": float(np.mean(scores)) if scores else None,
            "mean_true_quality": float(np.mean(quals)) if quals else None,
            "steps": res.steps,
            "seed": seed,
            # Which implementation produced this row, recorded in the row. A
            # table mixing tiny-LM fallback runs and real gpt2 runs with no way
            # to tell them apart is a table you cannot report from.
            "notes": notes,
        }
        store.insert_alignment_run(run_row)
        results.append(run_row | {"history": res.history})

    store.insert_completions(completions)

    base_scores = [r["reward_score"] for r in completions
                   if r["model_variant"] == "base" and r["reward_score"] is not None]
    base_quals = [r["true_quality"] for r in completions
                  if r["model_variant"] == "base" and r["true_quality"] is not None]

    # Win rate against the base model, PAIRED BY PROMPT. An unpaired comparison
    # of two means over 60 prompts is dominated by which prompts happened to be
    # easy; pairing removes the prompt effect, which is the largest source of
    # variance here.
    by_prompt_base = {r["prompt_id"]: r["reward_score"] for r in completions
                      if r["model_variant"] == "base"}
    for row in results:
        beta = row["beta"]
        mine = {r["prompt_id"]: r["reward_score"] for r in completions
                if r["model_variant"] == "dpo" and r["beta"] == beta}
        wins = [1.0 if mine[k] > by_prompt_base.get(k, -np.inf) else 0.0 for k in mine]
        row["win_rate_vs_base"] = float(np.mean(wins)) if wins else None

    summary = {
        "reference": {
            "sft_steps": sft_steps,
            "sft_history": sft_hist,
            "mean_reward_model_score": float(np.mean(base_scores)) if base_scores else None,
            "mean_true_quality": float(np.mean(base_quals)) if base_quals else None,
            "vocab_size": len(vocab),
        },
        "runs": results,
        "n_gen_prompts": len(gen_prompts),
        "completions_written": len(completions),
        "loss_monotonicity_check": verify_loss_decreases_with_margin(),
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "alignment.json").write_text(json.dumps(summary, indent=2))
    return summary


def _deployed_score_fn():
    """Score through the REAL `POST /score` handler, in process.

    Not by calling `RewardHead.score` directly. The point is that the number in
    the report is the number the service returns — including the artifact it
    resolved, the featuriser it applied and the audit row it wrote. A helper
    that reproduces the arithmetic would pass while the endpoint was broken.

    `TestClient` here rather than `httpx` against a live uvicorn so the offline
    pipeline needs no second process. Both go through the same handler.
    """
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app, raise_server_exceptions=False)

    def score(text: str) -> float | None:
        if not text.strip():
            return None
        r = client.post("/score", json={"text": text, "policy_name": "reward_tfidf"})
        return None if r.status_code >= 400 else float(r.json()["reward"])

    return score


def _completion_rows(prompts, prompt_ids, texts, variant, beta, score_fn, quality_fn,
                     experiment_id):
    from shared.preprocess import response_length

    rows = []
    for prompt, text in zip(prompts, texts):
        rows.append(
            {
                "prompt_id": prompt_ids.get(prompt, ""),
                "prompt": prompt,
                "model_variant": variant,
                "beta": beta,
                "text": text,
                "reward_score": score_fn(text),
                "true_quality": quality_fn(text),
                "tokens": response_length(text),
                "run_id": experiment_id,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Run DPO at several betas and persist the results.")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--pairs", type=int, default=2400)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.05, 0.1, 0.5])
    ap.add_argument("--sft-steps", type=int, default=400)
    ap.add_argument("--dpo-steps", type=int, default=300)
    ap.add_argument("--gen-prompts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.offline:
        raise SystemExit(
            "the real path runs TRL's DPOTrainer on gpt2 and needs a GPU runtime.\n"
            "Call train_dpo_trl() from a notebook, or pass --offline to run the "
            "self-contained tiny-LM fallback."
        )

    from train.data import load_preferences

    ds = load_preferences(True, n_pairs=args.pairs, seed=args.seed)
    summary = run_alignment(
        ds,
        betas=tuple(args.betas),
        sft_steps=args.sft_steps,
        dpo_steps=args.dpo_steps,
        n_gen_prompts=args.gen_prompts,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "base_mean_reward": summary["reference"]["mean_reward_model_score"],
                "base_mean_true_quality": summary["reference"]["mean_true_quality"],
                "runs": [
                    {
                        k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()
                        if k != "history"
                    }
                    for r in summary["runs"]
                ],
                "dpo_loss_monotone_in_margin": summary["loss_monotonicity_check"][
                    "monotone_decreasing"
                ],
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":  # pragma: no cover - a CLI entry point
    main()
