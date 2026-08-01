# Resources for AI Courses

A collection of teaching resources. The repository holds **four courses**, each
with its own tutorials and hands-on resources, plus a course-agnostic survey of
cloud deployment architectures:

| | What it is |
|---|---|
| 📘 **[Deep Learning](#deep-learning)** | Course tutorials + full-stack **project templates** (Streamlit + FastAPI + Supabase). |
| 📗 **[Intro to Machine Learning](#intro-to-machine-learning)** | Course tutorials **and single-tier project templates** (Streamlit + SQLite + pandas + scikit-learn) covering the classical ML toolkit. |
| 📕 **[Natural Language Processing](#natural-language-processing)** | Course lecture notes, exercises, and **starter skeletons** on the same three-cloud stack — infrastructure finished, the NLP layer left to build. |
| 📙 **[Reinforcement Learning](#reinforcement-learning)** | Course tutorials, exercises, and **complete agent templates** on a two-cloud, three-tier stack — train in PyTorch, serve in NumPy. |
| ☁️ **[Cloud Deployment](#cloud-deployment-survey)** | A survey of different deployment stacks (Azure, Vercel, Render, Railway, TF.js). Not tied to any one course. |

> Each course folder groups the tutorials and resources for one subject area. The
> **Cloud Deployment** folder sits alongside the courses because it is a general
> reference — it was first introduced within Deep Learning, but the deployment
> patterns it surveys apply broadly.

---

## Deep Learning

The [`Deep Learning/`](./Deep%20Learning) course pairs two subfolders that go
hand in hand to support each course topic:

- **[`Project Templates/`](./Deep%20Learning/Project%20Templates)** — **7 complete,
  forkable full-stack templates**, one per topic, all built on the same
  **three-cloud architecture** (Streamlit UI + FastAPI model API + Supabase data).
- **[`Tutorials/`](./Deep%20Learning/Tutorials)** — the matching topic tutorials
  (HTML + DOCX) that cover the underlying math and deep-learning concepts.

> **New here? Start with the three-cloud architecture below.** It is the
> canonical reference pattern; the six templates that follow it
> (*Income-Insight*, *See-Sense*, *Attend-It*, *Former-It*, *Fine-It*, *Gen-It*)
> reuse the same pattern with only the model box swapped.

---

### ⭐ three-cloud (reference architecture)
**Regress-It — Streamlit + FastAPI + Supabase**

The three-cloud reference template: an interactive linear-regression demo split
cleanly across three managed clouds. This is the pattern the other Streamlit
templates reuse, provided as a complete working, forkable template.

- **UI**: Streamlit on Streamlit Community Cloud (thin client, no model code)
- **API**: FastAPI on Render.com (PyTorch training, owns all writes)
- **Data**: Supabase Postgres (single source of truth, Row Level Security)
- **Highlights**: clean separation of concerns, service-role vs anon keys, RLS,
  pytest suite, model card, reusable across models
- **Cost**: Free tier on all three platforms

[→ View Project README](./Deep%20Learning/Project%20Templates/Topic_1_three-cloud/README.md) | [→ Tutorial](./Deep%20Learning/Project%20Templates/Topic_1_three-cloud/TUTORIAL.md)

---

### ⭐ income-insight
**Income-Insight — Tabular MLP Classifier on the same three clouds**

Reuses the three-cloud architecture with the model box swapped for a PyTorch MLP
+ sklearn preprocessing pipeline doing binary income classification (Adult-Income
shaped, synthetic data).

- **UI**: Streamlit on Streamlit Community Cloud (thin client, form built from `/schema`)
- **API**: FastAPI on Render.com (MLP + `ColumnTransformer`, owns all writes)
- **Data**: Supabase Postgres (datasets · runs · run_artifacts · predictions, RLS)
- **Highlights**: classification metrics (accuracy/precision/recall/F1/AUC),
  `/predict_batch`, `/schema`, and a fairness `/audit` endpoint; pytest suite
- **Cost**: Free tier on all three platforms

[→ View Project README](./Deep%20Learning/Project%20Templates/Topic_2_income-insight/README.md) | [→ Tutorial](./Deep%20Learning/Project%20Templates/Topic_1_three-cloud/TUTORIAL.md)

---

### ⭐ see-sense
**See-Sense — CNN Image Classifier + Grad-CAM on the same three clouds**

Reuses the three-cloud architecture with the model box swapped for a PyTorch CNN
doing image classification, plus **Grad-CAM** heatmaps that show *where the
network looked* (synthetic shape images).

- **UI**: Streamlit on Streamlit Community Cloud (thin client, image upload + Grad-CAM view)
- **API**: FastAPI on Render.com (CNN + Grad-CAM, owns all writes)
- **Data**: Supabase Postgres (datasets · runs · run_artifacts · image_metadata, RLS)
- **Highlights**: Grad-CAM explainability, "store the image hash not the pixels"
  privacy invariant, accuracy/macro-F1 metrics, `/predict_sample`; pytest suite
- **Cost**: Free tier on all three platforms

[→ View Project README](./Deep%20Learning/Project%20Templates/Topic_3_see-sense/README.md) | [→ Tutorial](./Deep%20Learning/Project%20Templates/Topic_1_three-cloud/TUTORIAL.md)

---

### ⭐ attend-it
**Attend-It — LSTM + Attention Sequence Classifier on the same three clouds**

Reuses the three-cloud architecture with the model box swapped for an LSTM with
**additive (Bahdanau) attention** classifying synthetic token sequences whose
label depends on a randomly-placed trigger token.

- **UI**: Streamlit on Streamlit Community Cloud (thin client, per-timestep attention view)
- **API**: FastAPI on Render.com (LSTM + attention, owns all writes)
- **Data**: Supabase Postgres (datasets · runs · run_artifacts · sequence_metadata, RLS)
- **Highlights**: attention-weight explainability, accuracy/macro-F1 metrics,
  "store the sequence hash not the tokens" privacy invariant; pytest suite
- **Cost**: Free tier on all three platforms

[→ View Project README](./Deep%20Learning/Project%20Templates/Topic_4_attend-it/README.md) | [→ Tutorial](./Deep%20Learning/Project%20Templates/Topic_1_three-cloud/TUTORIAL.md)

---

### ⭐ former-it
**Former-It — From-Scratch Transformer Encoder on the same three clouds**

Reuses the three-cloud architecture with the model box swapped for a
from-scratch tiny **Transformer encoder** (positional encoding + multi-head
self-attention) solving an algorithmic sequence task (palindrome detection).

- **UI**: Streamlit on Streamlit Community Cloud (thin client, per-head attention heatmaps)
- **API**: FastAPI on Render.com (Transformer encoder, owns all writes)
- **Data**: Supabase Postgres (datasets · runs · run_artifacts · sequence_metadata, RLS)
- **Highlights**: per-head N×N attention heatmaps, accuracy metric, hash-not-tokens
  privacy invariant; pytest suite
- **Cost**: Free tier on all three platforms

[→ View Project README](./Deep%20Learning/Project%20Templates/Topic_5_former-it/README.md) | [→ Tutorial](./Deep%20Learning/Project%20Templates/Topic_1_three-cloud/TUTORIAL.md)

---

### ⭐ fine-it
**Fine-It — Pretrain + Fine-Tune Char Transformer on the same three clouds**

Reuses the three-cloud architecture with the model box swapped for a causal
character **Transformer** trained in two phases — self-supervised next-character
**pretraining**, then classifier **fine-tuning** — to make the transfer gap
visible against a from-scratch baseline.

- **UI**: Streamlit on Streamlit Community Cloud (thin client, pretrained-vs-scratch bar + text generation)
- **API**: FastAPI on Render.com (char Transformer, LM + classifier heads, owns all writes)
- **Data**: Supabase Postgres (datasets · runs · run_artifacts · sequence_metadata, RLS)
- **Highlights**: transfer-learning demo (`/pretrain`, `/finetune`, `/generate`),
  temperature text sampling, hash-not-characters privacy invariant; pytest suite
- **Cost**: Free tier on all three platforms

[→ View Project README](./Deep%20Learning/Project%20Templates/Topic_6_fine-it/README.md) | [→ Tutorial](./Deep%20Learning/Project%20Templates/Topic_1_three-cloud/TUTORIAL.md)

---

### ⭐ gen-it
**Gen-It — Variational Autoencoder on the same three clouds**

Reuses the three-cloud architecture with the model box swapped for a
**variational autoencoder (VAE)** that learns a 2-D latent space over synthetic
images and then generates, reconstructs, and interpolates in it.

- **UI**: Streamlit on Streamlit Community Cloud (thin client, latent sliders + interpolation + 2-D scatter)
- **API**: FastAPI on Render.com (VAE encoder/decoder, owns all writes)
- **Data**: Supabase Postgres (datasets · runs · run_artifacts · image_metadata, RLS)
- **Highlights**: latent-space `/generate` `/reconstruct` `/interpolate` `/latent_scatter`,
  reconstruction-loss/KL/ELBO metrics, hash-not-pixels privacy invariant; pytest suite
- **Cost**: Free tier on all three platforms

[→ View Project README](./Deep%20Learning/Project%20Templates/Topic_7_gen-it/README.md) | [→ Tutorial](./Deep%20Learning/Project%20Templates/Topic_1_three-cloud/TUTORIAL.md)

---

## Intro to Machine Learning

The [`Intro to Machine Learning/`](./Intro%20to%20Machine%20Learning) course, like
Deep Learning, pairs two subfolders that go hand in hand to support each topic:

- **[`Project Templates/`](./Intro%20to%20Machine%20Learning/Project%20Templates)** —
  **7 complete, forkable single-tier templates**, one per topic, all built on the
  same lightweight stack: a **Streamlit** app running an in-process **scikit-learn**
  model, a local **SQLite** database (basic CRUD), and **pandas** for the analysis.
  No cloud backend to stand up — each one runs on Streamlit Community Cloud straight
  from a fork.
- **[`Tutorials/`](./Intro%20to%20Machine%20Learning/Tutorials)** — the matching
  **HTML tutorials and exercises** (a Tutorial + an Exercises file per topic) covering
  the underlying concepts and hands-on practice.

### Project Templates

Each template follows the same universal build pattern —
`raw CSV → SQLite (CRUD) → pandas + scikit-learn → Streamlit UI` — with deterministic
synthetic data, a pytest suite, and heavily commented code aimed at first-semester
students.

| Topic | Template | What it does |
|-------|----------|--------------|
| 1 | [SQL Foundations](./Intro%20to%20Machine%20Learning/Project%20Templates/Topic_1_sql-foundations/README.md) | Answer business questions over a synthetic SQLite database with basic CRUD and pandas + SQL. |
| 2 | [Data-Quality Dashboard](./Intro%20to%20Machine%20Learning/Project%20Templates/Topic_2_data-quality/README.md) | Profile a messy dataset, flag quality issues, and clean it — with a before/after quality score. |
| 3 | [Honest Forecast](./Intro%20to%20Machine%20Learning/Project%20Templates/Topic_3_honest-forecast/README.md) | Fit a linear-regression forecaster with an honest train/test split and held-out metrics. |
| 4 | [Mailguard](./Intro%20to%20Machine%20Learning/Project%20Templates/Topic_4_mailguard/README.md) | Classify messages with a Naive Bayes model and inspect which words drive each decision. |
| 5 | [ClassifierLab](./Intro%20to%20Machine%20Learning/Project%20Templates/Topic_5_classifier-lab/README.md) | Train and compare several scikit-learn classifiers head-to-head on the same task. |
| 6 | [LatentLens](./Intro%20to%20Machine%20Learning/Project%20Templates/Topic_6_latent-lens/README.md) | Segment shoppers with K-Means, visualise them with PCA, and mine association rules. |
| 7 | [FeatureForge](./Intro%20to%20Machine%20Learning/Project%20Templates/Topic_7_dvc-capstone/README.md) | Engineer features that measurably beat a baseline, wired into a reproducible DVC pipeline. |

### Tutorials

| Topic | Subject |
|-------|---------|
| 1 | SQL Basics — [Tutorial](./Intro%20to%20Machine%20Learning/Tutorials/Topic_1_SQL_Basics_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Tutorials/Topic_1_SQL_Basics_Exercises.html) |
| 2 | Data Quality — [Tutorial](./Intro%20to%20Machine%20Learning/Tutorials/Topic_2_Data_Quality_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Tutorials/Topic_2_Data_Quality_Exercises.html) |
| 3 | Linear Regression — [Tutorial](./Intro%20to%20Machine%20Learning/Tutorials/Topic_3_Linear_Regression_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Tutorials/Topic_3_Linear_Regression_Exercises.html) |
| 4 | Naive Bayes — [Tutorial](./Intro%20to%20Machine%20Learning/Tutorials/Topic_4_Naive_Bayes_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Tutorials/Topic_4_Naive_Bayes_Exercises.html) |
| 5 | Classification — [Tutorial](./Intro%20to%20Machine%20Learning/Tutorials/Topic_5_Classification_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Tutorials/Topic_5_Classification_Exercises.html) |
| 6 | Clustering — [Tutorial](./Intro%20to%20Machine%20Learning/Tutorials/Topic_6_Clustering_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Tutorials/Topic_6_Clustering_Exercises.html) |
| 7 | Data Version Control & Features — [Tutorial](./Intro%20to%20Machine%20Learning/Tutorials/Topic_7_DVC_Features_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Tutorials/Topic_7_DVC_Features_Exercises.html) |

> GitHub renders `.html` files as source. Clone or download the folder (or enable
> GitHub Pages) to view them as formatted pages in a browser.

---

## Natural Language Processing

The [`Natural Language Processing/`](./Natural%20Language%20Processing) course
pairs two subfolders, one set per topic:

- **[`Project Templates/`](./Natural%20Language%20Processing/Project%20Templates)** —
  **7 starter skeletons**, one per topic, on the same **three-cloud architecture**
  used in Deep Learning (Streamlit UI + FastAPI API + Supabase Postgres).
- **[`Tutorials/`](./Natural%20Language%20Processing/Tutorials)** — **lecture notes
  and exercises** (HTML) for each topic, covering the concepts behind the build.

A video walkthrough also accompanies each topic. The recordings are **not stored
in this repository** — video files would bloat every clone and fork — and are
distributed separately.

> **These templates are skeletons, not solutions.** Unlike the Deep Learning
> templates — which ship complete and working — each NLP template hands you a
> finished, deployable three-tier app with the NLP layer hollowed out. Every
> function in `api/nlp.py` is left to implement, and a `contract` test suite plus
> a `shared/schemas.py` contract define what "done" means. Topic 1 is the base
> template; Topics 2–7 reuse its exact layout with the middle box swapped.

### Project Templates

| Topic | Template | What you build |
|-------|----------|----------------|
| 1 | [TokenForge](./Natural%20Language%20Processing/Project%20Templates/Topic_1_tokenforge/README.md) | A text-preprocessing and tokenization service — at least two tokenization algorithms, compared side by side. **Start here: this is the base template.** |
| 2 | [Classify-It](./Natural%20Language%20Processing/Project%20Templates/Topic_2_classify-it/README.md) | A text-classification service, built twice — TF-IDF + logistic regression, then a transformer — returning a label, an actionable probability, and a record of every answer. |
| 3 | [GenText](./Natural%20Language%20Processing/Project%20Templates/Topic_3_gentext/README.md) | A controllable text-generation service on a pretrained decoder transformer, with human evaluation across scored quality dimensions. |
| 4 | [MoodLens](./Natural%20Language%20Processing/Project%20Templates/Topic_4_moodlens/README.md) | A sentiment and **aspect-based** sentiment service, with per-slice metrics rather than one headline number. |
| 5 | [EntityFinder](./Natural%20Language%20Processing/Project%20Templates/Topic_5_entityfinder/README.md) | A named-entity-recognition service where two models produce **spans** and a human review queue writes corrections back. |
| 6 | [TagWise](./Natural%20Language%20Processing/Project%20Templates/Topic_6_tagwise/README.md) | A part-of-speech tagging service, with error analysis over the word classes the model actually confuses. |
| 7 | [AskMyDocs](./Natural%20Language%20Processing/Project%20Templates/Topic_7_askmydocs/README.md) | A **retrieval-augmented** question-answering service — embeddings, chunking, and vector search on Supabase Postgres with **pgvector**. |

Every template ships with a `MODEL_CARD.md` to fill in, a `render.yaml` for the
API deploy, Supabase migrations, and a pytest suite split into `contract` and
`network` markers so the contract tests run offline.

### Tutorials

| Topic | Subject |
|-------|---------|
| 1 | Text Preprocessing and Tokenization — [Lecture Notes](./Natural%20Language%20Processing/Tutorials/Topic_1_-_Text_Preprocessing_and_Tokenization/Lecture_Notes_Topic_1_Text_Preprocessing_and_Tokenization.html) · [Exercises](./Natural%20Language%20Processing/Tutorials/Topic_1_-_Text_Preprocessing_and_Tokenization/Exercises_Topic_1_Text_Preprocessing_and_Tokenization.html) |
| 2 | Natural Language Understanding — [Lecture Notes](./Natural%20Language%20Processing/Tutorials/Topic_2_-_Natural_Language_Understanding/Lecture_Notes_Topic_2_Natural_Language_Understanding.html) · [Exercises](./Natural%20Language%20Processing/Tutorials/Topic_2_-_Natural_Language_Understanding/Exercises_Topic_2_Natural_Language_Understanding.html) |
| 3 | Natural Language Generation — [Lecture Notes](./Natural%20Language%20Processing/Tutorials/Topic_3_-_Natural_Language_Generation/Lecture_Notes_Topic_3_Natural_Language_Generation.html) · [Exercises](./Natural%20Language%20Processing/Tutorials/Topic_3_-_Natural_Language_Generation/Exercises_Topic_3_Natural_Language_Generation.html) |
| 4 | Sentiment Analysis — [Lecture Notes](./Natural%20Language%20Processing/Tutorials/Topic_4_-_Sentiment_Analysis/Lecture_Notes_Topic_4_Sentiment_Analysis.html) · [Exercises](./Natural%20Language%20Processing/Tutorials/Topic_4_-_Sentiment_Analysis/Exercises_Topic_4_Sentiment_Analysis.html) |
| 5 | Named Entity Recognition — [Lecture Notes](./Natural%20Language%20Processing/Tutorials/Topic_5_-_Named_Entity_Recognition/Lecture_Notes_Topic_5_Named_Entity_Recognition.html) · [Exercises](./Natural%20Language%20Processing/Tutorials/Topic_5_-_Named_Entity_Recognition/Exercises_Topic_5_Named_Entity_Recognition.html) |
| 6 | Part-of-Speech Tagging — [Lecture Notes](./Natural%20Language%20Processing/Tutorials/Topic_6_-_Part_of_Speech_Tagging/Lecture_Notes_Topic_6_Part_of_Speech_Tagging.html) · [Exercises](./Natural%20Language%20Processing/Tutorials/Topic_6_-_Part_of_Speech_Tagging/Exercises_Topic_6_Part_of_Speech_Tagging.html) |
| 7 | Language Modeling, Embeddings and RAG — [Lecture Notes](./Natural%20Language%20Processing/Tutorials/Topic_7_-_Language_Modeling_Embeddings_and_RAG/Lecture_Notes_Topic_7_Language_Modeling_Embeddings_and_RAG.html) · [Exercises](./Natural%20Language%20Processing/Tutorials/Topic_7_-_Language_Modeling_Embeddings_and_RAG/Exercises_Topic_7_Language_Modeling_Embeddings_and_RAG.html) |

> GitHub renders `.html` files as source. Clone or download the folder (or enable
> GitHub Pages) to view them as formatted pages in a browser.

---

## Reinforcement Learning

The [`Reinforcement Learning/`](./Reinforcement%20Learning) course pairs two
subfolders, one set per topic:

- **[`Project templates/`](./Reinforcement%20Learning/Project%20templates)** —
  **6 complete, forkable agent templates**, one per topic, plus a bare
  [`agent-template-base/`](./Reinforcement%20Learning/Project%20templates/agent-template-base/README.md)
  scaffold they all reuse.
- **[`Tutorials/`](./Reinforcement%20Learning/Tutorials)** — the matching
  **tutorials and exercises** (HTML), with an
  [index page](./Reinforcement%20Learning/Tutorials/index.html) linking all twelve.

### Architecture — two clouds, three tiers

Unlike the three-cloud courses, the RL templates deploy on **two** managed
clouds but keep **three** tiers separate in the code:

- **Presentation**: Streamlit on Streamlit Community Cloud
- **Service**: FastAPI + Pydantic contracts — run under uvicorn locally and
  imported in-process in production (`SERVICE_MODE` is the only switch)
- **Data**: Supabase Postgres — every episode of every run is a row, so the
  learning-curve comparisons are a `GROUP BY`, not a screenshot
- **Training**: PyTorch, on a laptop or in Colab. **Never deployed.**

> **The no-PyTorch-in-serving rule.** `import torch` alone costs ~490 MB against
> Streamlit Community Cloud's 690 MB guarantee. So training exports weights to a
> NumPy `.npz` and the serving path evaluates the forward pass in NumPy — the
> whole deployed stack measures ~82 MB. `tests/test_no_torch.py` asserts
> `"torch" not in sys.modules` after importing the app, and the build fails if it
> ever does. See [`docs/no-torch.md`](./Reinforcement%20Learning/Project%20templates/agent-template-base/docs/no-torch.md).

Every template ships the same six standing endpoints (`/act`, `/rollout`,
`/policies`, `/runs`, `/healthz`, `/version`), Supabase migrations, committed
`.npz` policy artifacts, a GitHub Actions CI workflow, and a pytest + ruff gate.

### Project Templates

| Topic | Template | What it does |
|-------|----------|--------------|
| — | [agent-template-base](./Reinforcement%20Learning/Project%20templates/agent-template-base/README.md) | The bare scaffold — tiers, contracts, store, and the no-torch guard, with the agent left out. **Start here to see the pattern.** |
| 1 | [Lake Pilot](./Reinforcement%20Learning/Project%20templates/topic-1-lake-pilot/README.md) | Q-learning on a slippery FrozenLake, with a trained agent and a random one behind the same endpoint — what "learning from experience" actually buys. |
| 2 | [Policy Lab](./Reinforcement%20Learning/Project%20templates/topic-2-policy-lab/README.md) | Value iteration and Monte Carlo control on one 5×5 routing grid — the exact answer and the learned one, with the gap reported as a confidence interval. |
| 3 | [Gradient Works](./Reinforcement%20Learning/Project%20templates/topic-3-gradient-works/README.md) | Policy gradients on CartPole trained four ways (±baseline × on/off-policy), reporting the **variance of the gradient estimate**, not just the score. |
| 4 | [Control Bench](./Reinforcement%20Learning/Project%20templates/topic-4-control-bench/README.md) | An actor-critic bake-off — A2C on CartPole, PPO on Acrobot, SAC on continuous Pendulum — three agents behind one API contract. |
| 5 | [Search Arena](./Reinforcement%20Learning/Project%20templates/topic-5-search-arena/README.md) | A playable Connect Four service holding six agents (exhaustive, alpha–beta, beam, MCTS/UCT, revised MCTS, PUCT self-play) under an enforced node budget. |
| 6 | [Alignment Lab](./Reinforcement%20Learning/Project%20templates/topic-6-alignment-lab/README.md) | A reward model trained from human comparisons, base vs aligned outputs, a KL sweep, the point where optimising the proxy stops helping, and a multi-agent finale. |

### Tutorials

| Topic | Subject |
|-------|---------|
| 1 | Introduction to Reinforcement Learning — [Tutorial](./Reinforcement%20Learning/Tutorials/Tutorial-Topic-1-Introduction-to-Reinforcement-Learning.html) · [Exercises](./Reinforcement%20Learning/Tutorials/Exercises-Topic-1-Introduction-to-Reinforcement-Learning.html) |
| 2 | Markov Decision Processes and Monte Carlo Learning — [Tutorial](./Reinforcement%20Learning/Tutorials/Tutorial-Topic-2-Markov-Decision-Process-and-Monte-Carlo-Learning.html) · [Exercises](./Reinforcement%20Learning/Tutorials/Exercises-Topic-2-Markov-Decision-Process-and-Monte-Carlo-Learning.html) |
| 3 | Policy Gradient Methods — [Tutorial](./Reinforcement%20Learning/Tutorials/Tutorial-Topic-3-Policy-Gradient-Methods.html) · [Exercises](./Reinforcement%20Learning/Tutorials/Exercises-Topic-3-Policy-Gradient-Methods.html) |
| 4 | Actor-Critic Methods — [Tutorial](./Reinforcement%20Learning/Tutorials/Tutorial-Topic-4-Actor-Critic-Methods.html) · [Exercises](./Reinforcement%20Learning/Tutorials/Exercises-Topic-4-Actor-Critic-Methods.html) |
| 5 | Tree Search — [Tutorial](./Reinforcement%20Learning/Tutorials/Tutorial-Topic-5-Tree-Search.html) · [Exercises](./Reinforcement%20Learning/Tutorials/Exercises-Topic-5-Tree-Search.html) |
| 6 | RLHF and Multi-Agent Reinforcement Learning — [Tutorial](./Reinforcement%20Learning/Tutorials/Tutorial-Topic-6-RLHF-and-Multi-Agent-Reinforcement-Learning.html) · [Exercises](./Reinforcement%20Learning/Tutorials/Exercises-Topic-6-RLHF-and-Multi-Agent-Reinforcement-Learning.html) |

> GitHub renders `.html` files as source. Clone or download the folder (or enable
> GitHub Pages) to view them as formatted pages in a browser — or open
> [`Tutorials/index.html`](./Reinforcement%20Learning/Tutorials/index.html) locally.

---

## Cloud Deployment (survey)

The [`Cloud deployment models/`](./Cloud%20deployment%20models) folder is a
**course-agnostic survey** of different frontend/backend deployment stacks. Unlike
the Deep Learning templates (which all share the three-cloud pattern), these five
React projects each demonstrate a *different* platform combination — a menu of
deployment architectures to compare. They live at the top level because the
patterns apply across courses.

---

### 1. react-azure
**Full-Stack Todo Application on Azure**

A production-ready todo application built with React and FastAPI, deployed on Microsoft Azure.

- **Frontend**: React (TypeScript) on Azure Static Web Apps
- **Backend**: FastAPI (Python) on Azure Container Apps
- **Features**: CRUD operations, filtering, responsive design
- **Highlights**: Containerized deployment, free-tier Azure services
- **Cost**: ~$5/month (Azure Container Registry only)

[→ View Project README](./Cloud%20deployment%20models/react-azure/README.md) | [→ Tutorial](./Cloud%20deployment%20models/react-azure/TUTORIAL.md)

---

### 2. react-render
**Image Classification with Deep Learning**

A full-stack deep learning application for image classification using PyTorch/TensorFlow.

- **Frontend**: React on Render (Static Site)
- **Backend**: FastAPI with pre-trained ResNet model on Render
- **Features**: Image upload, real-time classification, confidence scores
- **Highlights**: Complete ML pipeline, educational deep learning tutorial
- **Cost**: Free tier available

[→ View Project README](./Cloud%20deployment%20models/react-render/README.md)

---

### 3. react-vercel
**Deep Learning Deployment with Vercel + Railway**

Image classification application demonstrating Vercel and Railway deployment.

- **Frontend**: React (Vite) on Vercel
- **Backend**: FastAPI with TensorFlow/MobileNetV2 on Railway
- **Features**: Image classification, top-5 predictions, automatic documentation
- **Highlights**: Hybrid cloud deployment, optimized for ML workloads
- **Cost**: Free tier available on both platforms

[→ View Project README](./Cloud%20deployment%20models/react-vercel/README.md)

---

### 4. react-vercel-render
**Vercel + Render Deployment Pattern**

Alternative deployment approach using Vercel for frontend and Render for ML backend.

- **Frontend**: React on Vercel
- **Backend**: FastAPI with TensorFlow on Render
- **Features**: Image classification, health monitoring, batch processing support
- **Highlights**: Cost-effective hybrid deployment, detailed deployment guide
- **Cost**: Free tier available

[→ View Project README](./Cloud%20deployment%20models/react-vercel-render/README.md)

---

### 5. react-local
**Browser-Based Deep Learning (No Backend Required)**

Handwritten digit recognition running entirely in the browser using TensorFlow.js.

- **Frontend**: React with TensorFlow.js
- **Backend**: None (client-side only)
- **Features**: Interactive drawing canvas, real-time prediction, model training in browser
- **Highlights**: No server costs, privacy-friendly, WebGL acceleration
- **Cost**: Free (static hosting only)

[→ View Project README](./Cloud%20deployment%20models/react-local/README.md)

---

## Quick Comparison

| Project | Frontend Platform | Backend Platform | ML Framework | Use Case | Complexity |
|---------|------------------|------------------|--------------|----------|------------|
| **three-cloud** ⭐ | Streamlit Cloud | Render (FastAPI) + Supabase | PyTorch | Linear Regression (Regress-It) | Medium |
| **income-insight** ⭐ | Streamlit Cloud | Render (FastAPI) + Supabase | PyTorch MLP + sklearn | Tabular Classification (Income-Insight) | Medium |
| **see-sense** ⭐ | Streamlit Cloud | Render (FastAPI) + Supabase | PyTorch CNN + Grad-CAM | Image Classification (See-Sense) | Medium |
| **attend-it** ⭐ | Streamlit Cloud | Render (FastAPI) + Supabase | PyTorch LSTM + Attention | Sequence Classification (Attend-It) | Medium |
| **former-it** ⭐ | Streamlit Cloud | Render (FastAPI) + Supabase | PyTorch Transformer Encoder | Algorithmic Sequences (Former-It) | Medium |
| **fine-it** ⭐ | Streamlit Cloud | Render (FastAPI) + Supabase | PyTorch Char Transformer | Pretrain + Fine-Tune (Fine-It) | Medium |
| **gen-it** ⭐ | Streamlit Cloud | Render (FastAPI) + Supabase | PyTorch VAE | Generative Modelling (Gen-It) | Medium |
| **tokenforge** 📕 | Streamlit Cloud | Render (FastAPI) + Supabase | *yours to choose* | Tokenization (TokenForge) | Medium |
| **classify-it** 📕 | Streamlit Cloud | Render (FastAPI) + Supabase | TF-IDF + transformer | Text Classification (Classify-It) | Medium |
| **gentext** 📕 | Streamlit Cloud | Render (FastAPI) + Supabase | Pretrained decoder transformer | Text Generation (GenText) | Medium |
| **moodlens** 📕 | Streamlit Cloud | Render (FastAPI) + Supabase | *yours to choose* | Sentiment Analysis (MoodLens) | Medium |
| **entityfinder** 📕 | Streamlit Cloud | Render (FastAPI) + Supabase | *yours to choose* | Named Entities (EntityFinder) | Medium |
| **tagwise** 📕 | Streamlit Cloud | Render (FastAPI) + Supabase | *yours to choose* | POS Tagging (TagWise) | Medium |
| **askmydocs** 📕 | Streamlit Cloud | Render (FastAPI) + Supabase (pgvector) | Embeddings + retrieval | RAG Q&A (AskMyDocs) | Hard |
| **lake-pilot** 📙 | Streamlit Cloud | FastAPI (in-process) + Supabase | NumPy serving (tabular Q) | Q-Learning (Lake Pilot) | Medium |
| **policy-lab** 📙 | Streamlit Cloud | FastAPI (in-process) + Supabase | NumPy serving (tabular) | MDP + Monte Carlo (Policy Lab) | Medium |
| **gradient-works** 📙 | Streamlit Cloud | FastAPI (in-process) + Supabase | PyTorch → NumPy `.npz` | Policy Gradients (Gradient Works) | Medium |
| **control-bench** 📙 | Streamlit Cloud | FastAPI (in-process) + Supabase | PyTorch → NumPy `.npz` | Actor-Critic: A2C/PPO/SAC (Control Bench) | Hard |
| **search-arena** 📙 | Streamlit Cloud | FastAPI (in-process) + Supabase | NumPy + PUCT self-play net | Tree Search (Search Arena) | Hard |
| **alignment-lab** 📙 | Streamlit Cloud | FastAPI (in-process) + Supabase | PyTorch → NumPy `.npz` | RLHF + Multi-Agent (Alignment Lab) | Hard |
| React-Azure | Azure Static Web Apps | Azure Container Apps | None | Todo App | Medium |
| React-Render | Render | Render | PyTorch/TensorFlow | Image Classification | Medium |
| React-Vercel | Vercel | Railway | TensorFlow | Image Classification | Medium |
| React-Vercel-Render | Vercel | Render | TensorFlow | Image Classification | Medium |
| React-local | Any Static Host | None | TensorFlow.js | Digit Recognition | Easy |

> ⭐ = complete, working template. 📕 = starter skeleton — infrastructure
> finished, the NLP layer left to implement (hence *yours to choose* where the
> model is an open decision). 📙 = complete RL agent template — two clouds, three
> tiers, with training kept out of the deployed app (PyTorch trains, NumPy serves).

---

## Learning Objectives

By working through these projects, you will learn:

### Cloud Deployment
- Deploy React applications on multiple platforms (Azure, Vercel, Render)
- Deploy FastAPI backends with containerization
- Configure environment variables and CORS
- Understand different cloud pricing models

### Full-Stack Development
- Build React frontends with modern hooks and state management
- Create RESTful APIs with FastAPI
- Handle file uploads and processing
- Implement proper error handling and validation

### Deep Learning Integration
- Load and serve pre-trained models (ResNet, MobileNetV2)
- Implement image preprocessing pipelines
- Run inference in production environments
- Deploy ML models in browsers with TensorFlow.js

### DevOps & Best Practices
- Git-based deployment workflows
- Environment-specific configurations
- Health checks and monitoring
- Cost optimization strategies

---

## Prerequisites

### Required Software
- **Node.js** (v14 or higher)
- **Python** (v3.8 or higher)
- **Git**
- **Docker** (for containerized deployments)

### Cloud Accounts (Free Tiers Available)
- **Azure Account** (for React-Azure project)
- **Vercel Account** (for Vercel projects)
- **Render Account** (for Render projects)
- **Railway Account** (for React-Vercel project)
- **GitHub Account** (for all projects)

---

## Getting Started

### Choose Your Learning Path

**Path 1: Start Simple** (Recommended for beginners)
1. Start with **react-local** (no backend, browser-only)
2. Move to **react-render** (add backend deployment)
3. Try **react-vercel** (learn multi-platform deployment)

**Path 2: Cloud Platform Focus**
1. Learn **react-azure** (Microsoft Azure ecosystem)
2. Compare with **react-vercel-render** (alternative platforms)

**Path 3: Deep Learning Focus**
1. Explore **react-local** (browser-based ML)
2. Scale up with **react-render** (server-based ML)
3. Deploy to **react-vercel** (production ML)

### General Setup Steps

Each project follows a similar workflow:

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Resources-for-AI-courses
   ```

2. **Navigate to a project**
   ```bash
   cd "Deep Learning/[project-name]"
   ```

3. **Follow the project's README**
   - Each project has detailed setup instructions
   - Includes local development and deployment guides
   - Contains troubleshooting sections

---

## Project Structure

```
Resources-for-AI-courses/
├── README.md                          # This file
│
├── Deep Learning/                     # 📘 Course: tutorials + project templates
│   ├── Project Templates/            # ⭐ Streamlit + FastAPI + Supabase templates
│   │   ├── Topic_1_three-cloud/      # ⭐ Reference architecture
│   │   │   ├── ui/                   # Streamlit thin client
│   │   │   ├── api/                  # FastAPI + PyTorch (Render)
│   │   │   ├── db/                   # Supabase migrations + seed
│   │   │   ├── shared/               # Pydantic contract + data gen
│   │   │   ├── tests/                # pytest suite
│   │   │   ├── README.md             # Project documentation
│   │   │   └── TUTORIAL.md           # Step-by-step guide
│   │   ├── Topic_2_income-insight/   # ⭐ Tabular classifier (same 3 clouds)
│   │   ├── Topic_3_see-sense/        # ⭐ Image classifier (same 3 clouds)
│   │   ├── Topic_4_attend-it/        # ⭐ LSTM + attention (same 3 clouds)
│   │   ├── Topic_5_former-it/        # ⭐ Transformer encoder (same 3 clouds)
│   │   ├── Topic_6_fine-it/          # ⭐ Pretrain + fine-tune char Transformer
│   │   └── Topic_7_gen-it/           # ⭐ Variational autoencoder (same 3 clouds)
│   └── Tutorials/                    # HTML + DOCX topic tutorials (math + DL)
│
├── Intro to Machine Learning/         # 📗 Course: templates + HTML tutorials
│   ├── Project Templates/            # Streamlit + SQLite + pandas + scikit-learn
│   │   ├── Topic_1_sql-foundations/  # SQLite CRUD + pandas SQL analytics
│   │   ├── Topic_2_data-quality/     # Profiling + cleaning + quality score
│   │   ├── Topic_3_honest-forecast/  # Linear regression, honest train/test
│   │   ├── Topic_4_mailguard/        # Naive Bayes message classifier
│   │   ├── Topic_5_classifier-lab/   # Compare scikit-learn classifiers
│   │   ├── Topic_6_latent-lens/      # K-Means + PCA + association rules
│   │   └── Topic_7_dvc-capstone/     # Feature engineering + DVC pipeline
│   └── Tutorials/                    # HTML tutorials + exercises (Topics 1-7)
│
├── Natural Language Processing/       # 📕 Course: starter skeletons + notes
│   ├── Project Templates/            # Same three-cloud stack, NLP layer left to build
│   │   ├── Topic_1_tokenforge/       # Base template — tokenization
│   │   │   ├── ui/                   # Streamlit thin client
│   │   │   ├── api/                  # FastAPI (Render) — nlp.py is yours
│   │   │   ├── db/                   # Supabase migrations + seed
│   │   │   ├── shared/               # Pydantic contract
│   │   │   ├── tests/                # pytest (contract + network markers)
│   │   │   ├── README.md             # Project documentation
│   │   │   └── MODEL_CARD.md         # Model card to fill in
│   │   ├── Topic_2_classify-it/      # Text classification
│   │   ├── Topic_3_gentext/          # Controllable text generation
│   │   ├── Topic_4_moodlens/         # Sentiment + aspect-based sentiment
│   │   ├── Topic_5_entityfinder/     # NER + human review queue
│   │   ├── Topic_6_tagwise/          # Part-of-speech tagging
│   │   └── Topic_7_askmydocs/        # RAG Q&A (Supabase pgvector)
│   └── Tutorials/                    # Lecture notes + exercises (Topics 1-7)
│
├── Reinforcement Learning/            # 📙 Course: agent templates + tutorials
│   ├── Project templates/            # Streamlit + FastAPI + Supabase, NumPy serving
│   │   ├── agent-template-base/      # The bare scaffold every topic reuses
│   │   │   ├── ui/                   # Streamlit thin client
│   │   │   ├── api/                  # FastAPI service — NumPy only, never torch
│   │   │   ├── train/                # PyTorch training tier — never deployed
│   │   │   ├── envs/                 # Environment definitions
│   │   │   ├── policies/             # Exported .npz policy artifacts (committed)
│   │   │   ├── db/                   # Supabase migrations + seed
│   │   │   ├── shared/               # Pydantic contract, config, store
│   │   │   ├── tests/                # pytest suite (incl. the no-torch guard)
│   │   │   └── docs/no-torch.md      # Why serving must not import PyTorch
│   │   ├── topic-1-lake-pilot/       # Q-learning on slippery FrozenLake
│   │   ├── topic-2-policy-lab/       # Value iteration vs Monte Carlo control
│   │   ├── topic-3-gradient-works/   # Policy gradients + gradient variance
│   │   ├── topic-4-control-bench/    # A2C · PPO · SAC behind one contract
│   │   ├── topic-5-search-arena/     # Connect Four: minimax → MCTS → PUCT
│   │   └── topic-6-alignment-lab/    # RLHF, KL sweep, multi-agent
│   └── Tutorials/                    # HTML tutorials + exercises (Topics 1-6)
│
└── Cloud deployment models/           # ☁️ Course-agnostic deployment survey
    ├── react-azure/                  # Azure Static Web Apps + Container Apps
    │   ├── frontend/                 # React TypeScript app
    │   ├── backend/                  # FastAPI with Docker
    │   ├── README.md                 # Project documentation
    │   └── TUTORIAL.md               # Step-by-step guide
    ├── react-render/                 # Render (frontend + ML backend)
    ├── react-vercel/                 # Vercel + Railway
    ├── react-vercel-render/          # Vercel + Render
    └── react-local/                  # Browser-based ML (TensorFlow.js)
```

---

## Common Technologies

### Frontend Stack
- **React** - UI library
- **Vite** - Build tool (some projects)
- **Create React App** - Build tool (some projects)
- **TypeScript** - Type safety (Azure project)
- **Axios** - HTTP client

### Backend Stack
- **FastAPI** - Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **Python-multipart** - File upload handling

### Machine Learning
- **TensorFlow** - Deep learning framework
- **PyTorch** - Deep learning framework
- **TensorFlow.js** - Browser-based ML
- **Pre-trained Models** - ResNet, MobileNetV2

### DevOps & Deployment
- **Docker** - Containerization
- **GitHub Actions** - CI/CD (Azure project)
- **Azure CLI** - Azure deployments
- **Vercel CLI** - Vercel deployments

---

## Cost Considerations

All projects can be deployed on **free tiers**:

| Platform | Free Tier | Limitations |
|----------|-----------|-------------|
| Streamlit Community Cloud | Unlimited public apps | Sleeps after inactivity |
| Supabase | 500 MB database, 2 projects | Pauses after 1 week idle |
| Azure Static Web Apps | 100 GB bandwidth/month | Sufficient for learning |
| Azure Container Apps | 180,000 vCPU-seconds/month | Good for demos |
| Vercel | 100 GB bandwidth | Hobby projects |
| Render | 750 hours/month | Sleeps after inactivity |
| Railway | $5 free credit/month | Limited resources |

**Cost-Saving Tips:**
- Use free tiers for learning and testing
- Deploy during development, tear down when done
- Monitor usage in platform dashboards
- Use serverless for low-traffic apps

---

## Troubleshooting

### Common Issues Across Projects

**CORS Errors:**
- Verify backend CORS configuration includes frontend URL
- Check that API endpoint URLs are correct
- Ensure both frontend and backend are running

**API Connection Failed:**
- Confirm backend is running and accessible
- Check environment variables are set correctly
- Verify firewall/network settings

**Build Failures:**
- Clear caches: `npm cache clean --force`
- Delete `node_modules` and reinstall
- Check Node/Python versions

**Deployment Issues:**
- Review platform-specific logs
- Verify all environment variables are set
- Check that all files are committed to Git

For project-specific issues, consult individual README files.

---

## Learning Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [TensorFlow.js Guide](https://www.tensorflow.org/js)
- [Azure Documentation](https://docs.microsoft.com/azure)
- [Vercel Documentation](https://vercel.com/docs)
- [Render Documentation](https://render.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Supabase Documentation](https://supabase.com/docs)

### Tutorials & Courses
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [React Official Tutorial](https://react.dev/learn)

---

## Contributing

This is an educational repository. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make improvements (code, documentation, examples)
4. Submit a pull request

**Ideas for contributions:**
- Add new deployment platforms (AWS, GCP)
- Improve error handling
- Add authentication examples
- Create video tutorials
- Add database integration examples

---

## License

MIT License - Free for educational and commercial use.

---

## Support & Contact

For questions and support:
- Check individual project README files
- Review troubleshooting sections
- Open an issue on GitHub

---

## Acknowledgments

These projects demonstrate modern web development and cloud deployment best practices.

**Happy Learning & Building!** 🚀

---

*Last Updated: July 31, 2026*
