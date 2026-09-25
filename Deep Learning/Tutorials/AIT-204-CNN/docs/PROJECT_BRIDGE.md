# Project Bridge: From This Lab to the Topic 3 Project

The Topic 3 project ("See-Sense") accepts an uploaded image, classifies it,
returns a Grad-CAM overlay, and logs the prediction without keeping the image.
This lab builds a working, local version of every model-side piece. This page
shows where each piece goes when you build the real thing.

## What carries over unchanged

| Lab code | In the project |
|---|---|
| `SmallResNet` and `ResidualBlock` | Your model definition. Widen or deepen it to improve accuracy. |
| `build_train_transform` / `build_eval_transform` | Your data pipeline. Serving must use the eval transform. |
| `train.py` (`run_epoch`, scheduler, checkpoint with `run_id`) | Your training script. The `run_id` becomes `served_by_run_id`. |
| `compute_cam` / `GradCAM` | The Grad-CAM overlay returned by the API. |
| `Predictor.predict` | The body of the `POST /predict` handler. |
| `to_db_record` | The exact row you insert into the Supabase `predictions` table. |
| `api.py` | Your FastAPI service: multipart upload, JSON response. |
| `app.py` | Your Streamlit front end. |

## What the project adds

- **Deployment.** Host the API (the project uses Render's free tier) and configure `SEESENSE_CHECKPOINT` or load weights from storage. Handle cold starts: show an `st.spinner` and a friendly error after a 30 second timeout.
- **Supabase.** Create the `predictions` table and insert `to_db_record(result)` after each prediction. Keep the service-role key on the server only. The Streamlit "Recent Classifications" tab reads with the read-only anon key.
- **A stronger model.** Try the ideas below and report what helped, with numbers.
- **Full-dataset training.** Use `--subset 0` and more epochs (the lab defaults are sized for class time).

## Contract to preserve

`POST /predict` receives `multipart/form-data` with a `file` field and returns:

```json
{
  "sha256": "9f86d0...",
  "label": "horse",
  "confidence": 0.91,
  "top5": [{"label": "horse", "prob": 0.91}, "..."],
  "gradcam_png_b64": "iVBORw0KGgo...",
  "served_by_run_id": "3fa9c1de"
}
```

The stored row contains only `sha256`, `label`, `top5`, and `served_by_run_id`.
`tests/test_part6_predict.py::test_db_record_never_contains_image_data` enforces
this. Keep that test in your project.

## Ideas to improve on the lab baseline

Choose changes you can justify with a table like the Part 7 table.

1. Train on all 50,000 images for 30 or more epochs with cosine annealing.
2. Widen the network (`widths=(64, 128, 256)`) or add a second block per stage. Recompute the parameter count with `conv_params` first.
3. Transfer learning (tutorial section 14): fine-tune a pretrained ResNet-18 on upscaled inputs, and compare it with your from-scratch model.
4. Add warmup to the LR schedule (tutorial section 13).
5. Investigate failures with Grad-CAM. Find three confidently wrong predictions and explain each one.

## Known limitations to discuss in your write-up

- CIFAR-10 images are 32x32. The API downsizes every upload, so real photos lose most of their detail, and objects outside the ten classes still get a confident label.
- Grad-CAM at `layer3` is an 8x8 map, upsampled. It shows where the model looked, not a precise object outline.
- A hash lets you deduplicate and audit without storing the image, but it does not identify what the image contained.
