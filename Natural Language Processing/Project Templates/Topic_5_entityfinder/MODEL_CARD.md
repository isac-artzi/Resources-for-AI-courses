# Model Card — EntityFinder

> Fill this in as you build. It is rendered by the Model Card tab, so whatever
> you write here is what a user of your service sees. Delete every instruction
> line (the ones in blockquotes) before you submit.

## What this service does

> Two sentences, in plain language, for an analyst who is not an engineer.

## Entity types

| Type | What counts | What does not | Example we get wrong |
|------|-------------|---------------|----------------------|
| PER | | | |
| ORG | | | |
| LOC | | | |
| MISC | | | |

> The third and fourth columns are the ones a reader needs. "ORG: organisations"
> tells them nothing; "ORG: companies and institutions, but not the buildings
> they occupy, so 'the London office' is a LOC to us" tells them what to expect.

## Training data

> Name the corpus, cite it, and give the split sizes and label distribution you
> counted yourself. Include the share of tokens tagged O, and the token-level
> accuracy a model would get by predicting O everywhere — one line, and it
> explains your evaluation choices better than a paragraph would.

## Intended use

> What is this appropriate for? Name at least one concrete use it supports.

## Out of scope

> Where it should not be trusted. Be specific. "It is not perfect" is not an
> answer. The domain gap is the honest headline here: a model trained on
> decades-old newswire has seen the organisations and place names that were in
> the news then. Run it on text from the domain you actually care about,
> measure the drop, and put that number in this section.

## Models compared

| Model | What it sees | Entity F1 | Precision | Recall | Where it fails |
|-------|--------------|-----------|-----------|--------|----------------|
| CRF baseline | | | | | |
| Transformer | | | | | |

> These are ENTITY-level scores, computed by exact match on
> (start, end, type). If the number you paste here came from a token-level
> report, it is a different and larger number measuring a different and easier
> task, and every comparison below it is wrong.

## Per-type scores

| Type | Precision | Recall | F1 | Support |
|------|-----------|--------|----|---------|
| | | | | |

> The aggregate is dominated by whichever type is most frequent. This table is
> where the useful finding usually lives.

## Error analysis

> Three categories, one worked example each, with the actual text, the gold
> span, and what your model returned:
>
> 1. **Boundary errors** — right type, wrong extent. Show one, and say what the
>    entity-level score did to you for it.
> 2. **Type confusions** — right extent, wrong type. Which pair confuses your
>    model most, and why?
> 3. **Unseen entities** — a name the model has never encountered. What did it
>    do, and how confident was it while doing it?

## Decoding rules

| Situation | What we do | Why |
|-----------|-----------|-----|
| `I-X` with no entity open | | |
| `I-Y` continuing an open `X` | | |
| Entity confidence from token confidences | | |
| Documents longer than the model's maximum length | | |

> These are choices, not facts, and they change your recall. Write down what you
> chose so the next person can reproduce your numbers.

## Confidence and the review threshold

> What threshold did you set, and what happened at that setting: how many
> predictions landed in the queue, and how many of them turned out to be wrong?
> That second number is the one that tells you whether the threshold is doing
> its job. Note plainly that a softmax score is not a probability of being
> correct — a model can be wrong at 0.99, and it will be most confidently wrong
> on entities that look like ones it saw in training.

## Human review

> Who reviews, what they see, and what happens to their decision. Say explicitly
> that corrections are stored alongside the original prediction and never
> overwrite it, and say what you plan to do with the accumulated corrections.

## Known limitations

> At least three, from your own testing. Include the ones you found by feeding
> the service text it was not designed for.

## Privacy

> What is stored, what is not, and why. Explain the sha256 decision in your own
> words, and be honest about what the entities table still contains: the
> surface strings of the names your service found. If you enabled the context
> column, say so here and say why the tradeoff was worth it.

## Versions

| Version | Date | What changed |
|---------|------|--------------|
| entityfinder-v1 | | Initial build |
