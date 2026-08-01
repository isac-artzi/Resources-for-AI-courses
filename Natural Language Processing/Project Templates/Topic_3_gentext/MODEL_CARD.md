# Model Card — GenText

> Fill this in as you build. It describes a system that writes text on demand,
> so it is the document that tells someone whether they can trust the output.
> Delete every instruction line (the ones in blockquotes) before you submit.

## What this service does

> Two sentences, in plain language, for a content person who is not an engineer.

## Intended use

> What is this appropriate for? Name at least one concrete use it supports.

## Out of scope

> Where it should not be trusted. Be specific. "It is not perfect" is not an
> answer; "it invents product specifications that sound plausible, so it must not
> be used for anything a customer will act on" is.

## Base model and adaptation

| | |
|---|---|
| Base model (Hugging Face id) | |
| Adaptation method | full fine-tune / LoRA / prefix / none |
| Why this base model | |
| Why this method | |
| Published checkpoint | |
| Model version string | |

> "Why this base model" is a real question. Size, licence, tokenizer, and the
> memory ceiling of a free instance are all legitimate reasons. "It was in the
> tutorial" is not.

## Training corpus

| | |
|---|---|
| Source | |
| Domain | |
| Sentences after filtering | |
| Filters applied, in order | |
| sha256 of the filtered corpus | |

> The sentence count is the one after filtering, and the assignment sets a floor
> of 20,000. Say what the count was before, too — the gap is informative.
>
> Also state what you checked the corpus for. A generation model reproduces what
> it was trained on, including names, addresses and slurs.

## Decoding strategies exposed

| Strategy | What it does | When it is the right choice | What it costs |
|----------|--------------|-----------------------------|---------------|
| Greedy | | | |
| Beam search | | | |
| Temperature sampling | | | |
| top-k sampling | | | |
| top-p (nucleus) sampling | | | |

## Evaluation

### Automatic

| Metric | Value | On what |
|--------|-------|---------|
| Perplexity, base model | | held-out split |
| Perplexity, after adaptation | | held-out split |
| distinct-1 / distinct-2 by strategy | | generated outputs |

> Report the held-out perplexity before and after. If the fine-tune made it
> worse, say so — that is a result, and a small corpus with a high learning rate
> produces it reliably.

### Human

| | |
|---|---|
| Outputs rated | |
| Raters (labels, not names) | |
| Rubric dimensions | |
| Scale | |
| Exact agreement | |
| Mean absolute difference | |
| Cases differing by 2 or more | |

> The mean score is the least interesting number here. Take the outputs where the
> two of you differed by 2 or more, read them side by side, and write about what
> the rubric failed to pin down.

## Failure modes observed

> At least two, each with a generated example you actually produced, the settings
> that produced it, and what you did about it.

| Failure mode | Example (abridged) | Settings | Mitigation |
|--------------|--------------------|----------|------------|
| | | | |
| | | | |

## Coherence over longer outputs

> Where does it hold together and where does it drift? Give the token count at
> which your outputs start to lose the thread.

## Known limitations

> At least three, from your own testing. Include the ones you found by feeding
> the service prompts it was not designed for.

## Privacy

> What is stored, what is not, and why. Explain the prompt-hashing decision in
> your own words, and say what the stored OUTPUT could still reveal — a
> generation can echo its training data, and the output column is stored in full.

## Versions

| Version | Date | Base model | What changed |
|---------|------|-----------|--------------|
| gentext-v1 | | | Initial build |
