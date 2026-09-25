# Instructor Guide

## Purpose

Students arrive at the Topic 3 project needing to build a CNN, explain it with
Grad-CAM, and ship it behind a FastAPI endpoint that respects a data-handling
policy. This lab rehearses each of those skills in one session, with tests that
give immediate feedback so you can circulate instead of debugging setup.

## Before class

1. **Pre-download CIFAR-10.** The dataset is about 170 MB, and the source server can be slow (about 20 minutes on a poor connection). Run `python -m seesense_lab.check` once, then share the resulting `data/` folder (USB, LMS, or shared drive), or ask students to run the command the night before.
2. **Verify the environment** with `LAB_TARGET=solutions pytest`. All 40 tests should pass.
3. **Decide whether to distribute `solutions/`.** For the student copy, delete that folder. Nothing in `seesense_lab/` depends on it.
4. **Optional:** train a reference checkpoint (`LAB_TARGET=solutions python -m seesense_lab.train`) so pairs who fall behind on Parts 1 to 4 can still do Parts 5 and 6 with a shared `checkpoints/model.pt`.

## Suggested timing (90 minutes)

| Min | Activity |
|---|---|
| 0-5 | Part 0: setup and device check. Pair up. |
| 5-15 | Part 1: shape arithmetic. Have pairs do the hand questions **before** coding. |
| 15-25 | Part 2: augmentation. |
| 25-40 | Part 3: residual block and network inspection. |
| 40-55 | Part 4: implement the scheduler, start training. Use the training time for Part 5 (train in one terminal, code in another). |
| 55-70 | Part 5: Grad-CAM by hand, then in the app. |
| 70-80 | Part 6: serving logic, API smoke test. |
| 80-90 | Part 7 experiments C (shortcut) and exit ticket. Assign A and B as homework. |

If time is short, cut Experiments A and B (assign as homework) and keep C. The shortcut experiment is the one students remember.

## Expected results

Measured on an Apple Silicon laptop with the default settings
(`--epochs 5 --subset 10000`); your numbers will vary by a few points.

| Run | Train acc | Test acc | Gap | Other |
|---|---|---|---|---|
| Baseline | 60.9% | 60.7% | +0.2 | about 2 minutes on Apple Silicon (MPS) |
| A. `--no-augment` | 68.3% | 63.1% | +5.2 | Higher test accuracy at 5 epochs, wider gap |
| B. `--no-scheduler` | 58.4% | 48.3% | +10.1 | Test accuracy swung 44% then 26% then 48% in epochs 3 to 5 |
| C. `--shortcut` | 63.8% | 51.0% | +12.7 | 98.8% of stamped test images predicted "horse" |

Read the *pattern*, not the exact values:

- **A (no augmentation):** at 5 epochs the un-augmented model is *ahead* on test accuracy, and its train/test gap is wider. This surprises students who expect augmentation to always help. Augmentation trades early speed for later generalization: the model cannot memorize a 10,000-image subset when every epoch shows different crops. Have them rerun with `--epochs 20` (about 8 minutes) to see the crossover, or assign it as homework.
- **B (constant LR):** the biggest effect of the three. Without decay the test metrics are erratic from epoch to epoch and end about 12 points lower, because the final epochs never settle. Compare the `lr` columns.
- **C (shortcut):** nearly every stamped test image is called "horse" (a clean model gives about 10%), and clean test accuracy drops. The model spent capacity on a corner patch instead of on horses. Grad-CAM makes the cause visible.

## Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| Part 1 off by one | Used true division or forgot the `+ 1` | Use `//`; re-derive with the 32/3/1/1 example |
| `conv_params` off by `c_out` | Forgot bias, or added one bias in total | One bias per **output channel** |
| Part 2 test fails on order | `ToTensor()` before `RandomCrop` | Crop and flip act on PIL images |
| Part 3 `body` "not defined" | Edited `body` instead of `forward` | Only `forward` has a TODO |
| Part 3 applies ReLU to the body only | `shortcut(x) + relu(body(x))` | ReLU wraps the **sum** |
| Part 4 scheduler flat | Called `scheduler.step()` per batch, or not at all | Provided `train.py` steps once per epoch; check `T_max` |
| Part 5 map all zeros | Forgot ReLU-then-normalize order, or divided before ReLU | ReLU first, then divide by the max |
| Part 5 NaN | No guard for `max == 0` | Return the zero map unchanged |
| Part 6 unstable predictions | Missing `model.eval()` | Add it in `Predictor.__init__` |
| `mps` error | Unsupported op on older PyTorch | `--device cpu` |
| Training very slow | CPU-only laptop | `--subset 5000 --epochs 3` |

## Discussion prompts

- **After Part 3:** the residual block adds `x` back before the ReLU. What if it were added after? (The output could be negative and the "identity when the body is zero" property is lost for negative inputs.)
- **After Part 5:** show the same image with the Grad-CAM for the top-1 and top-2 classes. Where do they differ?
- **After Experiment C:** "Your validation accuracy is 90%. Grad-CAM shows the corner. Ship it?" Push students to name what evidence would change their mind, and how they would fix the dataset, not the model.
- **Privacy:** why store a hash and not the image? What does the hash still let you do (deduplicate, audit, join to user-provided feedback), and what can it not do?
- **Distribution shift:** the API resizes any photo to 32x32. Upload a dog photo taken on a phone. Why is the confidence still high or low? Does confidence mean correctness?

## Grading (20 points)

| Item | Points |
|---|---|
| All tests pass (Parts 1 to 6): 1 point per two tests, rounded, up to 10 | 10 |
| Part 1 hand answers shown, with arithmetic | 2 |
| Part 7 table filled in from real runs | 3 |
| Two Grad-CAM screenshots, each with a specific interpretation | 3 |
| Experiment C paragraph: identifies a dataset problem, proposes a data fix | 2 |

Signals of copied work: identical Part 7 numbers across pairs (training is
seeded, but hardware differences change results slightly), or a Grad-CAM
interpretation that does not match the screenshot.

## Mapping to the Topic 3 exercises

| Lab item | Topic 3 Exercises question |
|---|---|
| Part 1 tests | Q1 (output size), Q2 (receptive field), Q3 (stride to halve) |
| Part 2 | Q4 (augmentation choice) |
| Part 3 | Q5 (strided conv vs max pool), Q10 (residual block) |
| Part 5, Experiment C | Q6 (Grad-CAM interpretation) |
| Part 6 | Q7 (BatchNorm at inference), Q8 (data-handling policy), Q9 (upload encoding) |

## Extending the lab

- Swap `SmallResNet` for a pretrained `torchvision` ResNet-18 to demonstrate transfer learning (tutorial section 14).
- Add a `--warmup` option to `build_optimizer_and_scheduler` and a matching test.
- Add a fourth Grad-CAM target layer (`layer2`) and compare heatmap resolution.
- Make the API write `to_db_record(result)` to a real Supabase table, as in the project.
