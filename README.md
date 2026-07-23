# Resources for AI Courses

A collection of teaching resources. The repository holds **two courses**, each
with its own tutorials and hands-on resources, plus a course-agnostic survey of
cloud deployment architectures:

| | What it is |
|---|---|
| 📘 **[Deep Learning](#deep-learning)** | Course tutorials + full-stack **project templates** (Streamlit + FastAPI + Supabase). |
| 📗 **[Intro to Machine Learning](#intro-to-machine-learning)** | Course tutorials covering the classical ML toolkit (SQL, data quality, regression, classification, clustering). |
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

The [`Intro to Machine Learning/`](./Intro%20to%20Machine%20Learning) course is a
set of self-contained **HTML tutorials and exercises** (a Tutorial + an Exercises
file per topic) covering the classical ML toolkit — no deployment templates,
just the concepts and hands-on practice:

| Topic | Subject |
|-------|---------|
| 1 | SQL Basics — [Tutorial](./Intro%20to%20Machine%20Learning/Topic_1_SQL_Basics_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Topic_1_SQL_Basics_Exercises.html) |
| 2 | Data Quality — [Tutorial](./Intro%20to%20Machine%20Learning/Topic_2_Data_Quality_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Topic_2_Data_Quality_Exercises.html) |
| 3 | Linear Regression — [Tutorial](./Intro%20to%20Machine%20Learning/Topic_3_Linear_Regression_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Topic_3_Linear_Regression_Exercises.html) |
| 4 | Naive Bayes — [Tutorial](./Intro%20to%20Machine%20Learning/Topic_4_Naive_Bayes_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Topic_4_Naive_Bayes_Exercises.html) |
| 5 | Classification — [Tutorial](./Intro%20to%20Machine%20Learning/Topic_5_Classification_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Topic_5_Classification_Exercises.html) |
| 6 | Clustering — [Tutorial](./Intro%20to%20Machine%20Learning/Topic_6_Clustering_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Topic_6_Clustering_Exercises.html) |
| 7 | Data Version Control & Features — [Tutorial](./Intro%20to%20Machine%20Learning/Topic_7_DVC_Features_Tutorial.html) · [Exercises](./Intro%20to%20Machine%20Learning/Topic_7_DVC_Features_Exercises.html) |

> GitHub renders `.html` files as source. Clone or download the folder (or enable
> GitHub Pages) to view them as formatted pages in a browser.

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
| React-Azure | Azure Static Web Apps | Azure Container Apps | None | Todo App | Medium |
| React-Render | Render | Render | PyTorch/TensorFlow | Image Classification | Medium |
| React-Vercel | Vercel | Railway | TensorFlow | Image Classification | Medium |
| React-Vercel-Render | Vercel | Render | TensorFlow | Image Classification | Medium |
| React-local | Any Static Host | None | TensorFlow.js | Digit Recognition | Easy |

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
├── Intro to Machine Learning/         # 📗 Course: HTML tutorials + exercises
│   ├── Topic_1_SQL_Basics_*.html
│   ├── Topic_2_Data_Quality_*.html
│   ├── Topic_3_Linear_Regression_*.html
│   ├── Topic_4_Naive_Bayes_*.html
│   ├── Topic_5_Classification_*.html
│   ├── Topic_6_Clustering_*.html
│   └── Topic_7_DVC_Features_*.html
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

*Last Updated: July 22, 2026*
