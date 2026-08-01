# Model Card — TagWise

> Fill this in as you build. It is rendered by the Model Card tab, so whatever
> you write here is what a user of your service sees. Delete every instruction
> line (the ones in blockquotes) before you submit.

## What this service does

> Two sentences, in plain language, for an analyst who is not an engineer. Say
> what a part-of-speech tag is and what someone would do with one.

## Intended use

> What is this appropriate for? Name at least one concrete downstream use it
> supports — a parser, an entity extractor, a search index that wants nouns.

## Out of scope

> Where it should not be trusted. Be specific. "It is not perfect" is not an
> answer; "it was built on edited written English, so it tags clinical
> abbreviations as proper nouns" is.

## Corpus

| | |
|---|---|
| Treebank | |
| Revision / release | |
| Tag set | |
| Train sentences / tokens | |
| Dev sentences / tokens | |
| Test sentences / tokens | |

> Paste the tag distribution of the training split here, and state the share of
> the most frequent tag. That share is the accuracy of a tagger that guesses one
> tag for everything, and it is the floor every number below has to clear.

## The two taggers

| | Baseline | Transformer |
|---|---|---|
| What it is | most-frequent-tag lookup + rules for unseen words | fine-tuned token classifier |
| Built from | | |
| Sees context? | no — a function of the word alone | yes |
| Hyperparameters | casing, tie-break, fallback default | base checkpoint, learning rate, epochs, batch size, seed |
| Build time | | |
| Model size on disk | | |
| Can you explain a single prediction? | yes, point at the table entry or the rule | |

## Unknown-word handling

> The baseline's rules, in the order they fire, in a table. One row per rule:
> the condition, the tag it produces, and an example word it fires on. Then the
> number that matters: what fraction of test tokens were unknown, and what your
> accuracy was on that subset versus on the rest. Nearly all of the baseline's
> remaining error lives in those tokens, and this is where you show it.

## Results

| Model | Token accuracy | Macro-F1 | Tag set | Split |
|-------|---------------|----------|---------|-------|
| Baseline | | | | test |
| Transformer | | | | test |

> Both rows must be the same split and the same tag set or the comparison means
> nothing. If the transformer lost, report that it lost and say why — a lookup
> table beating an undertrained transformer is a real and common result, and
> hiding it is worse than having it.

> Say something about the gap between accuracy and macro-F1 for each model, and
> name the tags responsible.

## Ambiguous word classes

> At least three, each with a worked example: the two sentences, the gold tags,
> what each model said, and the reason the lookup could not have been right
> about both. NOUN/VERB, ADJ/ADV, and DET/PRON are the usual suspects; use your
> own confusion matrix rather than that list.

| Word | Sentence A (gold) | Sentence B (gold) | Baseline | Transformer |
|------|-------------------|-------------------|----------|-------------|
| | | | | |

## Known limitations

> At least three, from your own testing. Include the ones you found by feeding
> the service text it was not designed for — a tweet, a chemical name, a line of
> code, a sentence with no punctuation.

## Privacy

> What is stored, what is not, and why. Explain the sha256 decision in your own
> words, and say why a tag sequence is safe to keep when the sentence is not.

## Versions

| Version | Date | What changed |
|---------|------|--------------|
| tagwise-v1 | | Initial build |
