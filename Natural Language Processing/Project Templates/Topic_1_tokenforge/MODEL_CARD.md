# Model Card — TokenForge

> Fill this in as you build. It is rendered by the Model Card tab, so whatever
> you write here is what a user of your service sees. Delete every instruction
> line (the ones in blockquotes) before you submit.

## What this service does

> Two sentences, in plain language, for an analyst who is not an engineer.

## Intended use

> What is this appropriate for? Name at least one concrete use it supports.

## Out of scope

> Where it should not be trusted. Be specific. "It is not perfect" is not an
> answer; "it drops emoji, so it is not suitable for social-media sentiment
> pipelines" is.

## Preprocessing pipeline

| Step | Applied by default? | What it removes | What that costs |
|------|--------------------|-----------------|-----------------|
| Lowercase | | | |
| Strip punctuation | | | |
| Remove stop words | | | |
| Remove digits | | | |
| Stem / lemmatize | | | |

> The last column is the one that matters. For each step, give a sentence whose
> meaning changes when the step runs.

## Tokenizers compared

| Tokenizer | Algorithm | Vocab size | Pieces on our test text | OOV rate |
|-----------|-----------|-----------|------------------------|----------|
| | | | | |
| | | | | |

## Known limitations

> At least three, from your own testing. Include the ones you found by feeding
> the service text it was not designed for.

## Privacy

> What is stored, what is not, and why. Explain the sha256 decision in your own
> words.

## Versions

| Version | Date | What changed |
|---------|------|--------------|
| tokenforge-v1 | | Initial build |
