# Model Card — AskMyDocs

> Fill this in as you build. It is rendered by the Model Card tab, so whatever
> you write here is what a user of your service sees. Delete every instruction
> line (the ones in blockquotes) before you submit.

## What this service does

> Two sentences, in plain language, for a knowledge-management lead who is not
> an engineer. Say that it answers from a specific document collection and cites
> what it used, not that it "leverages AI".

## Intended use

> What is this appropriate for? Name at least one concrete question it answers
> well, with the passage it retrieves to do it.

## Out of scope

> Where it should not be trusted. Be specific. "It is not perfect" is not an
> answer; "it retrieves 400-token passages, so a fact that requires reading two
> distant sections of the same contract will not be found" is.

## The document collection

| | Retrieval collection | Fine-tuning corpus |
|---|---|---|
| Name | | |
| Source | | |
| Domain | | |
| Documents | | |
| Total tokens | | |
| Licence / permission to use | | |

> The two columns must describe two different sets of documents. See the section
> below.

## Corpus separation

> The most important paragraph in this card. State how you kept the fine-tuning
> corpus disjoint from the retrieval collection, and give the evidence: the
> output of `check_corpus_disjointness` (exact duplicates, worst shingle
> Jaccard), and whether the API's 409 or the database's unique index ever fired
> during development.
>
> Then say, in your own words, why it matters — that a generator which memorised
> the documents it later retrieves answers correctly with retrieval switched off,
> so the with/without comparison flattens to nothing and measures your data
> handling rather than your pipeline.

## Chunking

| Parameter | Value | Why |
|-----------|-------|-----|
| Chunk size (tokens) | | |
| Overlap (tokens / percent) | | |
| Tokenizer used to count | | |
| Boundary rule (raw window / sentence-aware) | | |

> The "why" column is graded. "It was the default" is not a reason. Include at
> least one example of a question your chunk size got right and one it got wrong.

## Embedding model

| | |
|---|---|
| Model id | |
| Dimensions | |
| Normalised? | |
| Same model used for queries and for chunks? | |

> That last row looks trivial. It is the single most common cause of "retrieval
> returns unrelated passages", and it produces no error message at all.

## Generator

| | |
|---|---|
| Model id | |
| Local weights or hosted inference API? | |
| Fine-tuned? On what? | |
| Peak memory observed | |
| Decoding settings (temperature, sampling, max new tokens) | |

> Say which memory you actually observed, not which you hoped for. If you moved
> generation to a hosted API because the free plan could not hold the model, say
> so — that is an engineering decision with a justification, not a shortcut.

## Perplexity

| Split | Documents | Tokens scored | Perplexity |
|-------|-----------|---------------|------------|
| Held-out (fine-tuning corpus) | | | |
| Training split (optional, for contrast) | | | |

> Report the held-out number. If you also report the training number, the gap
> between them is the interesting part and you should say what it tells you.
>
> Two caveats to state explicitly: perplexity is not comparable across
> tokenizers, and a low perplexity says nothing about whether an answer is
> correct — a model can be confidently fluent and wrong.

## Retrieval quality

| k | Mean similarity at rank 1 | Mean similarity at rank k | Questions answered from the collection |
|---|--------------------------|---------------------------|----------------------------------------|
| 5 | | | |

> Pull these from the Retrieval Audit tab. A flat similarity curve across ranks
> means the retriever is not discriminating, and that is worth explaining.

## With and without retrieval

> At least ten questions answered both ways, and at least three cases where the
> retrieved passage prevented a hallucination — each with the passage quoted.
> Put the full table in the project report; put the three cases here.

| Question | Without retrieval said | With retrieval said | Passage that grounded it |
|----------|-----------------------|---------------------|--------------------------|
| | | | |
| | | | |
| | | | |

## Known limitations

> At least three, from your own testing. Include at least one case where a
> passage was retrieved and the generator ignored it, since that failure is
> invisible unless you look at the audit table.

## Privacy

> What is stored, what is not, and why. Cover both halves: questions are stored
> only as a sha256 hash, and passages are stored in full and are readable by
> anyone holding the anon key. Explain what that second fact means for the choice
> of document collection.

## Versions

| Version | Date | What changed |
|---------|------|--------------|
| askmydocs-v1 | | Initial build |
