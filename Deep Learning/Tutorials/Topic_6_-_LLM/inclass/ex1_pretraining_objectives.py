"""
AIT-204 · Topic 6 · In-class Exercise 1 — Four homework games: MLM, CLM, span corruption, contrastive
Time: ~15–20 min   |   Needs: torch

    MLM  (BERT)    : hide ~15% of tokens, predict them from BOTH sides       — loss on masked positions only
    CLM  (GPT)     : predict token t+1 from tokens 0…t                       — loss on every position
    span (T5)      : replace spans with sentinels, decode the missing spans  — encoder–decoder cross-entropy
    contrastive    : pull paired sentence vectors together, push others away — InfoNCE

Every objective builds its own (input, target) pair from raw text. No human labels anywhere.
Fill in every "TODO". The asserts check your code against the Part B hand calculations.
"""
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=3, sci_mode=False)

PAD, MASK = 0, 1            # special token ids; real vocabulary ids start at 2
IGNORE = -100               # the label value that cross_entropy skips


# ---------------------------------------------------------------------------
# TODO 1 — masked language modeling: the 80 / 10 / 10 rule
# ---------------------------------------------------------------------------
def mlm_corrupt(ids, vocab_size, mask_prob=0.15, gen=None):
    """ids: (B, T) token ids. Returns (inputs, labels).
    Select each position with probability mask_prob. For the selected positions:
        80% → replace with MASK      10% → replace with a random token      10% → leave unchanged
    labels = the ORIGINAL id at selected positions and IGNORE everywhere else."""
    selected = torch.rand(ids.shape, generator=gen) < mask_prob
    roll = torch.rand(ids.shape, generator=gen)                     # decides 80 / 10 / 10
    random_ids = torch.randint(2, vocab_size, ids.shape, generator=gen)
    inputs = ids.clone()
    # TODO 1a: labels — original ids where selected, IGNORE elsewhere (torch.where)
    labels = None  # ← your code here
    # TODO 1b: where selected and roll < 0.8 → MASK; where selected and 0.8 ≤ roll < 0.9 → random_ids
    pass  # ← your code here (two assignments into `inputs`)
    return inputs, labels


# ---------------------------------------------------------------------------
# TODO 2 — causal language modeling: shift by one
# ---------------------------------------------------------------------------
def clm_shift(ids):
    """ids: (B, T). Returns (inputs, labels), both (B, T−1): the label at position t is the token at t+1."""
    # TODO 2: one line of slicing
    return None, None  # ← your code here


# ---------------------------------------------------------------------------
# TODO 3 — T5 span corruption with sentinel tokens
# ---------------------------------------------------------------------------
def span_corrupt(tokens, spans):
    """tokens: list of words. spans: sorted, non-overlapping (start, length) pairs.
    Each span is replaced by one sentinel in the input; the target lists each sentinel followed by
    the words it hid, then one final closing sentinel.
        ["The","quick","brown","fox"], [(1, 2)]  →  input  ["The","<X>","fox"]
                                                    target ["<X>","quick","brown","<Y>"]"""
    sentinels = ["<X>", "<Y>", "<Z>", "<W>"]
    inp, tgt, pos = [], [], 0
    for k, (start, length) in enumerate(spans):
        # TODO 3: copy the untouched words before this span into inp, add sentinel k to inp;
        #         add sentinel k and then the hidden words to tgt; move pos past the span.
        pass  # ← your code here
    inp += tokens[pos:]
    tgt.append(sentinels[len(spans)])
    return inp, tgt


# ---------------------------------------------------------------------------
# TODO 4 — contrastive learning: InfoNCE with in-batch negatives
# ---------------------------------------------------------------------------
def info_nce(z_a, z_b, tau=0.05):
    """z_a, z_b: (N, d). Row i of z_a and row i of z_b are two views of the same sentence (a positive pair).
    Every other row in the batch is a negative. Returns (loss, similarity matrix / tau)."""
    # TODO 4: L2-normalize both (F.normalize, dim=-1), build the N×N cosine-similarity matrix,
    #         divide by tau, then cross-entropy where the correct "class" for row i is column i.
    logits = loss = None  # ← your code here
    return loss, logits


if __name__ == "__main__":
    g = torch.Generator().manual_seed(0)
    V = 1000

    # ---- TODO 1 --------------------------------------------------------------
    ids = torch.randint(2, V, (200, 128), generator=g)
    inputs, labels = mlm_corrupt(ids, V, gen=g)
    sel = labels != IGNORE
    frac_sel = sel.float().mean().item()
    frac_mask = (inputs[sel] == MASK).float().mean().item()
    frac_kept = (inputs[sel] == ids[sel]).float().mean().item()
    print(f"selected {frac_sel:.3f}   of those: [MASK] {frac_mask:.3f}   unchanged {frac_kept:.3f}")
    assert torch.equal(inputs[~sel], ids[~sel]), "unselected positions must not change"
    assert torch.equal(labels[sel], ids[sel]), "labels hold the ORIGINAL ids"
    assert abs(frac_sel - 0.15) < 0.01 and abs(frac_mask - 0.80) < 0.03 and abs(frac_kept - 0.10) < 0.03
    logits = torch.randn(200, 128, V, generator=g)
    loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=IGNORE)
    print(f"MLM loss over {int(sel.sum())} of {labels.numel()} positions = {loss:.3f}  (a uniform guess scores ln V = {torch.log(torch.tensor(float(V))):.3f})")
    print("✓ TODO 1: 15% selected, split 80/10/10, loss only where labels ≠ −100")

    # ---- Part B check: MLM loss when the model gives p = 0.5 and p = 0.125 to the two masked words
    hand = -(torch.log(torch.tensor(0.5)) + torch.log(torch.tensor(0.125))) / 2
    print(f"Part B.2 by hand: {hand:.3f}")
    assert abs(hand.item() - 1.386) < 1e-3

    # ---- TODO 2 --------------------------------------------------------------
    sent = torch.tensor([[11, 12, 13, 14, 15, 16, 17, 18, 19]])        # nine "words"
    x, y = clm_shift(sent)
    print("CLM inputs", x.tolist(), "\nCLM labels", y.tolist())
    assert x.shape == y.shape == (1, 8) and torch.equal(x[:, 1:], y[:, :-1])
    print(f"✓ TODO 2: one 9-token sentence → {y.numel()} CLM targets, versus about {0.15 * 9:.2f} MLM targets")

    # ---- TODO 3 --------------------------------------------------------------
    words = "The quick brown fox jumps over the lazy dog".split()
    inp, tgt = span_corrupt(words, [(1, 2), (5, 3)])
    print("T5 input :", " ".join(inp), "\nT5 target:", " ".join(tgt))
    assert inp == ["The", "<X>", "fox", "jumps", "<Y>", "dog"]
    assert tgt == ["<X>", "quick", "brown", "<Y>", "over", "the", "lazy", "<Z>"]
    print("✓ TODO 3: text in, text out — the encoder reads 6 tokens, the decoder writes 8")

    # ---- TODO 4 --------------------------------------------------------------
    # Three unit vectors chosen so the anchor's cosine similarities are exactly [0.9, 0.1, 0.1] (Part B.3).
    anchor = torch.tensor([[1.0, 0.0, 0.0]])
    c = 0.1
    cands = torch.tensor([[0.9, (1 - 0.81) ** 0.5, 0.0],
                          [c, 0.0, (1 - c * c) ** 0.5],
                          [c, 0.0, -(1 - c * c) ** 0.5]])
    for tau in (0.1, 1.0):
        sims = F.normalize(anchor, dim=-1) @ F.normalize(cands, dim=-1).T / tau
        print(f"tau = {tau}:  logits {[round(v, 2) for v in sims[0].tolist()]}  →  loss for the anchor = {F.cross_entropy(sims, torch.tensor([0])):.4f}")
    z_a = torch.randn(8, 16, generator=g)
    loss_twins, lg = info_nce(z_a, z_a + 0.05 * torch.randn(8, 16, generator=g))
    loss_rand, _ = info_nce(z_a, torch.randn(8, 16, generator=g))
    print(f"InfoNCE, true twins: {loss_twins:.3f}    unrelated pairs: {loss_rand:.3f}    (a uniform guess scores ln 8 = 2.079; confident and wrong scores worse)")
    assert lg.shape == (8, 8) and loss_twins < 0.1 < loss_rand
    print("✓ TODO 4: twins on the diagonal → low loss; unrelated pairs → high loss")

    # QUESTION (handout, feeds DQ 1 and your Concepts tab): for each of the four objectives, write one row —
    # what the model sees, what it must produce, where the loss is computed, and the task it is best at.
