# Model Card — MoodLens

> Fill this in as you build. It is rendered by the Model Card tab, and its
> "Documented failures" section is rendered by the Bias Audit tab, so whatever
> you write here is what a user of your service sees. Delete every instruction
> line (the ones in blockquotes) before you submit.

## What this service does

> Two sentences, in plain language, for an analyst who is not an engineer. Say
> what it takes in and what it gives back, including the aspect breakdown.

## Intended use

> What is this appropriate for? Name at least one concrete use it supports, with
> the human step that goes with it.

## Out of scope

> Where it must not be trusted. Be specific and be operational. "It is not
> perfect" is not an answer. "It has not been evaluated on messages under twenty
> words, so it must not be used to route support tickets" is.
>
> Sentiment analysis attracts a particular kind of misuse: reading a score about
> a *text* as a fact about a *person*. State plainly that this service scores
> writing, not writers, and that it infers nothing about anyone's intent,
> motives, or character.

## Model

| | Transformer | TF-IDF baseline |
|---|---|---|
| Base model / features | | |
| Training data and size | | |
| Split (train / validation / test, seed) | | |
| Max sequence length and truncation | | n/a |
| Key hyperparameters | | |
| Wall-clock training time | | |

## Aspects, and how they were defined

| Aspect | What counts as this aspect | What was hard to place |
|--------|---------------------------|------------------------|
| | | |
| | | |
| | | |

> These are a modelling decision, not a property of the data. The third column
> is the honest one: write down the sentences you had to make a judgement call
> on, and the rule you settled on. Also say where the aspect labels came from —
> annotated by you (how many, by whom, with what guideline?) or shipped with the
> corpus.

## Held-out performance

| Model | n | Accuracy | Macro precision | Macro recall | Macro F1 |
|-------|---|----------|-----------------|--------------|----------|
| Fine-tuned transformer | | | | | |
| TF-IDF baseline | | | | | |

| Aspect | Precision | Recall | F1 | n evaluated |
|--------|-----------|--------|----|-------------|
| | | | | |
| | | | | |
| | | | | |

> Both models, always. If the transformer does not clearly beat TF-IDF on your
> split, say so and explain why you think that is — a long, balanced, well-
> written corpus is friendly territory for a linear model over word counts.

## Where the transformer beats the baseline

> Three input types, one worked example each: the text, what each model said,
> and what the correct answer was. Negation, long-range context and unfamiliar
> vocabulary are the usual candidates; use your own examples, not these.

## Calibration

> Which method (none, Platt, isotonic), fitted on which split, and what the
> reliability curve looked like before and after. If you did not calibrate, say
> so here and make sure the service reports `calibrated: false` — an
> uncalibrated score dressed up as a probability is the failure this section
> exists to prevent.

## Bias audit

| Slice | Bucket | n | Accuracy | Macro F1 | Observed or inferred? |
|-------|--------|---|----------|----------|----------------------|
| | | | | | |
| | | | | | |

> At least two slices. For each, say whether the attribute was measured or
> guessed. A slice you inferred with another model audits that model too, and
> the reader needs to know which numbers are which.
>
> Then say what you conclude, including the null result if that is what you
> found: "a 4-point gap on buckets of 800 and 90 is not something we can
> distinguish from noise" is a better sentence than a confident wrong one.

## Documented failures

> THIS SECTION IS RENDERED BY THE BIAS AUDIT TAB AND IT IS GRADED. At least
> three failures you found by testing your own service. Keep the heading exactly
> as it is or the tab will not find the section.
>
> Each row: the input, what the model said, what is correct, the ethical risk it
> illustrates, and the use limitation that follows from it. The limitation must
> be something a person could actually enforce.

| # | Input | Model said | Correct | Ethical risk | Use limitation this implies |
|---|-------|-----------|---------|--------------|-----------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

> Sarcasm and negation are the obvious places to look. So is short, noisy text,
> and so is any writing that does not look like the corpus you trained on —
> including English written by people who write it differently from the reviewers
> in your training data, which is where a failure stops being a curiosity and
> starts being a fairness problem.

## Human review triggers

> Two or more conditions under which a prediction must not be acted on without a
> person looking at it — a confidence threshold, a disagreement between the
> document label and the aspect breakdown, a slice known to be weak, an input
> type the model was never evaluated on. Write them as rules someone could
> implement, not as good intentions.

## Privacy

> What is stored, what is not, and why. Explain the sha256 decision in your own
> words, and say what an audit can and cannot reconstruct as a result — that
> trade-off is deliberate and you should be able to defend both halves of it.

## Known limitations

> At least three, from your own testing, beyond the failures listed above.

## Versions

| Version | Date | Base model | What changed |
|---------|------|-----------|--------------|
| moodlens-v1 | | | Initial build |
