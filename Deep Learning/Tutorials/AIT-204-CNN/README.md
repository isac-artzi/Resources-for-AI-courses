# AIT-204 CNN Lab: Build the See-Sense Pipeline

A guided in-class exercise that prepares you for the **Topic 3 project** (the
"See-Sense" image-classification product). In about 90 minutes you build, test,
and run every model-side piece of that product on a small scale:

| Lab part | You build | Topic 3 tutorial section | Project piece it rehearses |
|---|---|---|---|
| 1. Shape arithmetic | Output-size, parameter-count, and receptive-field formulas | 2, 4, 7 | Reasoning about your architecture |
| 2. Data and augmentation | The CIFAR-10 training transform | 12 | Data pipeline |
| 3. Residual network | A residual block and a small ResNet | 10, 11 | Model definition |
| 4. Training | SGD plus a cosine learning-rate schedule | 13 | Training run and `run_id` |
| 5. Grad-CAM | The Grad-CAM heatmap computation | 15 | Explainability overlay |
| 6. Serving logic | SHA-256 hashing, top-5, `eval()` mode, no-raw-bytes policy | 16 | `POST /predict` and the Supabase row |
| 7. Experiments | Augmentation, LR schedule, and a shortcut-learning demo | 12, 13, 15 | Debugging with Grad-CAM |

Each part has a **TODO** in the code and a **test file** that tells you when you
are done. The plumbing (data loaders, FastAPI service, Streamlit app) is
provided so you can spend your time on the ideas.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m seesense_lab.check          # prints your device, downloads CIFAR-10 (~170 MB)
pytest                                # every test fails until you finish the TODOs
```

Then open **[docs/LAB_GUIDE.md](docs/LAB_GUIDE.md)** and work through the parts in order.

```bash
pytest tests/test_part1_shapes.py     # check one part at a time
python -m seesense_lab.train          # train once Parts 1-4 pass (a few minutes)
streamlit run seesense_lab/app.py     # classify images with Grad-CAM overlays
uvicorn seesense_lab.api:app          # the FastAPI service, as in the project
```

## Repository layout

```
seesense_lab/          Your working code (contains the TODOs)
  shapes.py            Part 1  TODO
  data.py              Part 2  TODO
  model.py             Part 3  TODO
  train.py             Part 4  TODO
  gradcam.py           Part 5  TODO
  predict.py           Part 6  TODO
  api.py               Given: FastAPI service (POST /predict, multipart upload)
  app.py               Given: Streamlit front end
  check.py, device.py  Given: setup check and device selection
tests/                 One test file per part
solutions/             Instructor reference implementations (see below)
docs/
  LAB_GUIDE.md         Student handout: step-by-step instructions and questions
  INSTRUCTOR_GUIDE.md  Timing, common mistakes, discussion prompts, grading
  PROJECT_BRIDGE.md    How each piece maps onto the Topic 3 project
```

## For instructors

`solutions/` holds complete versions of every file with a TODO. Run the full
pipeline or the test suite against the reference code with:

```bash
LAB_TARGET=solutions pytest
LAB_TARGET=solutions python -m seesense_lab.train
```

To keep the solutions from students, delete `solutions/` from the copy you
distribute. See [docs/INSTRUCTOR_GUIDE.md](docs/INSTRUCTOR_GUIDE.md).

## Requirements

Python 3.10 to 3.13 and the packages in `requirements.txt`. A GPU is optional:
the lab auto-selects CUDA, then Apple Silicon (MPS), then CPU, and the default
settings (10,000 training images, 5 epochs) take about 2 minutes on an Apple Silicon GPU and a few minutes on a laptop CPU.

## Related course material

- [Topic 3 tutorial](../Topic_3_-_CNN/Tutorial_Topic_3_Convolutional_Neural_Networks.html)
- [Topic 3 exercises](../Topic_3_-_CNN/Exercises_Topic_3_Convolutional_Neural_Networks.html)
