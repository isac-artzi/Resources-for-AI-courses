# Model Card — Classify-It

> Fill this in as you build. It is rendered by the Model Card tab, so whatever
> you write here is what a user of your service sees. Delete every instruction
> line (the ones in blockquotes) before you submit.

## What this service does

> Two sentences, in plain language, for an operations analyst who is not an
> engineer. Name the two labels and say what a prediction is used for.

## Intended use

> What is this appropriate for? Name at least one concrete decision it supports,
> and say whether a human reviews the output.

## Out of scope

> Where it should not be trusted. Be specific. "It is not perfect" is not an
> answer; "it was trained on English-language support tickets from one product,
> so it should not be used on survey free text from a different domain" is.
> Single-label and binary: say plainly that it assigns exactly one of two labels
> and cannot express "both" or "neither".

## The data

| | |
|---|---|
| Corpus | |
| Public source | |
| Rows | |
| Labels | |
| Class balance | |
| Held-out split | |
| De-duplicated? | |

> The class balance row is not a formality. If it is 90/10, every accuracy figure
> below has to be read against a 0.90 baseline that does nothing.

### Label definitions

| Label | What an annotator had to see to assign it |
|-------|-------------------------------------------|
| | |
| | |

## Models compared

| | Baseline | Transformer |
|---|---|---|
| Approach | TF-IDF + logistic regression | fine-tuned encoder |
| Checkpoint / features | | |
| Key hyperparameters | | |
| Training time | | |
| Artifact size | | |
| Median inference latency | | |
| Model version string | | |

## Held-out results

> Same split, same positive label, for both models. All four numbers.

| Model | Accuracy | Precision | Recall | F1 | Support (positive) |
|-------|---------|-----------|--------|-----|--------------------|
| Baseline | | | | | |
| Transformer | | | | | |

> One paragraph reading these numbers. Which metric matters most for this
> product and why? If the transformer wins on accuracy but not on F1, say so —
> that is a finding, not something to hide.

## Where the transformer wins

> The Build Steps ask for at least three input types where the transformer beats
> the baseline, each with one worked example. Fill in the text, both predictions,
> the true label, and one sentence on why the encoder had the advantage.

| # | Input type | Example text | Baseline said | Transformer said | True label | Why |
|---|-----------|--------------|---------------|------------------|-----------|-----|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

> And, honestly: name at least one case where the baseline wins or ties. There
> almost always is one, and finding it is worth more than another table.

## Calibration

> Which method (Platt / isotonic / none), what data it was fitted on, and the
> evidence. Bucket held-out predictions by predicted probability and compare
> predicted against observed frequency, before and after. If you did not
> calibrate, write "uncalibrated" here and remove the word "calibrated" from any
> claim you make about the probability.

## Known limitations

> At least three, from your own testing. Include the ones you found by feeding
> the service text it was not designed for: another language, sarcasm, an empty
> message, something twice as long as your truncation length.

## Privacy

> What is stored, what is not, and why. Explain the sha256 decision in your own
> words, and note that the predictions table is readable with the anon key —
> which is only acceptable because it holds no input text.

## Versions

| Version | Date | Model kind | What changed |
|---------|------|-----------|--------------|
| classify-it-v1 | | | Initial build |
