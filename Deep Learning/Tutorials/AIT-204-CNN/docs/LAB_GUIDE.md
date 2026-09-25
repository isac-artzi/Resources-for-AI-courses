# Lab Guide: Build the See-Sense Pipeline

**Time:** about 90 minutes in class, working in pairs. **Goal:** finish every
model-side piece of the Topic 3 project on a small scale, so the project itself is
mostly deployment work.

Keep the [Topic 3 tutorial](../../Topic_3_-_CNN/Tutorial_Topic_3_Convolutional_Neural_Networks.html)
open. Each part names the section you need.

## How the lab works

- Every part has one or more `# TODO n.n` markers in `seesense_lab/`.
- Every part has a test file in `tests/`. **A part is done when its tests pass.**
- Do the parts in order. Later parts import earlier ones.
- Do not edit the tests. If a test fails, the test is right and your code is wrong.
  Read the assertion message, and compare shapes and values.

```bash
pytest tests/test_part1_shapes.py   # replace with the part you are on
pytest                              # everything
```

## Part 0: Setup (5 min)

```bash
python -m seesense_lab.check
```

You should see your Python and torch versions, the device that will be used
(`cuda`, `mps`, or `cpu`), and `CIFAR-10 ready: 50,000 train / 10,000 test images`.

**Record:** which device did you get? ______

## Part 1: Shape arithmetic (10 min)

File: `seesense_lab/shapes.py`. Tutorial sections 2, 4, and 7.

Implement `conv_out_size`, `conv_params`, and `receptive_field` with plain Python
arithmetic. The tests check your answers against real `nn.Conv2d` layers.

**Before you code, work these out by hand:**

1. `Conv2d(3, 32, kernel_size=3, stride=1, padding=1)` on a 32x32 input. Output size? ______
2. The same layer with `stride=2`. Output size? ______
3. Parameters in `Conv2d(3, 64, 3)`? ______ (The tutorial says 1792. Show the arithmetic.)
4. Receptive field of three stacked 3x3 convolutions? ______
5. Why does the parameter count not depend on the input's height and width?

Run `pytest tests/test_part1_shapes.py`. If your hand answers and the tests disagree, find out why.

## Part 2: Data and augmentation (10 min)

File: `seesense_lab/data.py`. Tutorial section 12.

Implement `build_train_transform`. Then look at what it does:

```python
from PIL import Image
from torchvision import datasets
from seesense_lab.data import build_train_transform

img, _ = datasets.CIFAR10("./data", train=True)[0]
t = build_train_transform()
print(t(img).mean(), t(img).std())   # roughly 0 and 1 after normalization
```

**Answer:**

1. Why do `RandomCrop` and `RandomHorizontalFlip` come *before* `ToTensor()`?
2. Would `RandomVerticalFlip` be a good augmentation for CIFAR-10? For a satellite-image classifier? Why might the answer differ?
3. The evaluation transform (`build_eval_transform`) has no augmentation. Why must it be deterministic?

## Part 3: The residual network (15 min)

File: `seesense_lab/model.py`. Tutorial sections 10 and 11.

`ResidualBlock.body` and the network wiring are provided. You write one line:
the block's `forward`.

After the tests pass, inspect the network:

```python
from seesense_lab.model import SmallResNet, count_parameters
m = SmallResNet()
print(count_parameters(m))
for name, p in m.named_parameters():
    print(f"{name:35s} {tuple(p.shape)}")
```

**Answer:**

1. Layer by layer, what is the spatial size of the feature map, from the 32x32 input to `layer3`? (Use your Part 1 function.)
2. `layer2` and `layer3` use `stride=2` instead of max pooling. Which parameters does that add, and why is this the "learnable down-sampler" from the tutorial?
3. Why do the convolutions use `bias=False` when a `BatchNorm2d` follows them?
4. `test_zero_body_makes_block_an_identity_relu` is Topic 3 Exercises question 10. In your own words: why does the residual connection make deep networks easier to train?

## Part 4: Training (15 min)

File: `seesense_lab/train.py`. Tutorial section 13.

Implement `build_optimizer_and_scheduler`. Then train:

```bash
python -m seesense_lab.train --epochs 5 --subset 10000
```

Watch three things: the `lr` column (does it follow a half-cosine?), the
train/test accuracy `gap`, and the throughput line at the end.

**Record:** final test accuracy ______  time ______  images/second ______

Then read `run_epoch` (it is provided). Notice `model.train(training)` and
`torch.set_grad_enabled(training)`.

**Answer:** what would go wrong if evaluation ran with `model.train()` still on?
Name the layer that changes behavior.

## Part 5: Grad-CAM (15 min)

File: `seesense_lab/gradcam.py`. Tutorial section 15.

Implement `compute_cam`. The provided `GradCAM` class hooks `layer3`, runs a
forward and backward pass for one class, and calls your function. Work the
first test (`test_compute_cam_by_hand`) out on paper first: it uses a 2x2 grid
and two feature maps.

Launch the app once Parts 1 to 6 pass:

```bash
streamlit run seesense_lab/app.py
```

Upload a few photos: an airplane, a cat, a truck, and something CIFAR-10 has no class for.

**Answer:**

1. Why is there a ReLU in the Grad-CAM formula? What does a negative value mean?
2. `layer3` is 8x8. What does the code do to get a 32x32 overlay, and what does that cost in precision?
3. For a wrong prediction, does the heatmap look reasonable? What does it show?

## Part 6: Serving logic (15 min)

File: `seesense_lab/predict.py`. Tutorial section 16.

Three small TODOs: `sha256_hex`, `top_k`, and `model.eval()`. Together they
decide whether your service is correct and whether it respects the data policy.

```python
from seesense_lab.predict import Predictor, to_db_record
p = Predictor.from_checkpoint("checkpoints/model.pt")
result = p.predict(open("some_image.jpg", "rb").read())
print({k: v for k, v in result.items() if k != "gradcam_png_b64"})
print(to_db_record(result))
```

**Answer:**

1. `test_prediction_is_stable_across_calls` catches a missing `eval()`. Explain the failure in terms of BatchNorm statistics. (Exercises question 7.)
2. `to_db_record` stores a SHA-256 hash and metadata, never image bytes. Give one privacy reason and one operational reason. (Exercises question 8.)
3. Why is a hash still useful without the image? What can you do with it?
4. Start the API with `uvicorn seesense_lab.api:app`, then send an image: `curl -F "file=@some_image.jpg" http://127.0.0.1:8000/predict`. Why is the upload `multipart/form-data` rather than JSON? (Exercises question 9.)

## Part 7: Experiments (remaining time)

Run these from the same starting point (`--epochs 5 --subset 10000`) and fill
in the table. Change **one thing at a time**. Each run overwrites
`checkpoints/`, so copy anything you want to keep first.

| Run | Command flags | Train acc | Test acc | Gap | Notes |
|---|---|---|---|---|---|
| Baseline | (none) | | | | |
| A. No augmentation | `--no-augment` | | | | |
| B. Constant LR | `--no-scheduler` | | | | |
| C. Shortcut | `--shortcut` | | | | see below |

**Experiment A.** Predict first: will removing augmentation raise or lower *train*
accuracy? *Test* accuracy? The gap? Then check. The result may surprise you at 5
epochs. Explain it, then rerun both settings with `--epochs 20` as homework and
report whether the ranking changes.

**Experiment B.** Compare the last few epochs' loss with and without the schedule.

**Experiment C: shortcut learning.** `--shortcut` stamps a small magenta square in
the top-left corner of every *horse* training image. After training it prints the
percentage of test images, all now stamped, that the model calls "horse".

1. Is that percentage near the 10% a clean model would give?
2. Load the checkpoint in the app. Upload a non-horse image with a magenta
   square in the top-left corner (draw one in any image editor). What does Grad-CAM show?
3. This is Topic 3 Exercises question 6, made real. If a horse classifier's Grad-CAM
   consistently highlights a watermark, is that a model bug or a dataset bug? What would you change?

## Deliverables

Submit:

1. Your `seesense_lab/` folder with every test passing (`pytest` output pasted or screenshotted).
2. A one-page `LAB_NOTES.md` with: your hand answers from Part 1, the Part 7 table, and one short paragraph on what Experiment C taught you.
3. Two Grad-CAM screenshots from the app: one correct prediction, and one wrong or shortcut-driven prediction, each with a one-sentence interpretation.

## Exit ticket (answer without notes)

1. A `Conv2d(64, 128, 3, stride=2, padding=1)` receives 16x16 input. Output shape and parameter count?
2. Name two reasons a residual connection helps.
3. Why does the FastAPI service call `model.eval()`?
4. What does the Supabase row contain, and what must it never contain?

*Answers: (1) 8x8 with 128 channels, 73,856 parameters. (2) Identity is free when the body outputs zero; gradients flow through the additive path. (3) BatchNorm must use running statistics and Dropout must be off. (4) Hash plus label, top-5, and run ID; never raw image bytes.*

## Stuck?

- `NotImplementedError` means you have a TODO left in that function.
- A test that passes alone but fails in `pytest` usually means an earlier part is wrong.
- `mps` errors on Apple Silicon: rerun with `--device cpu`.
- Slow download: ask your instructor for the pre-downloaded `data/` folder.
