# Stage Age

Baseline project for three-class age-group prediction from TA healthy ultrasound images.

## Task

- Images: `/home/szdx/LNX/data/TA/Healthy/Images`
- Metadata: `/home/szdx/LNX/data/TA/characteristics.xlsx`
- Matching key: image names follow `anon_{Number}_{view}.png`, where `Number` matches the healthy-group `Number` column in the Excel file.
- Classes:
  - `18-44`: `18 <= age < 45`
  - `45-59`: `45 <= age < 60`
  - `60-100`: `60 <= age < 101`
- Subjects younger than 18 are excluded.
- Splits are subject-level stratified splits to avoid leakage across views of the same subject.

## Environment

Use the existing `us` conda environment:

```bash
conda activate us
```

Or run commands through:

```bash
conda run -n us python ...
```

## Inspect Data

From this project directory:

```bash
PYTHONPATH=src python -m stage_age.inspect_data --config configs/resnet18.json
```

To write the manifest:

```bash
PYTHONPATH=src python -m stage_age.inspect_data \
  --config configs/resnet18.json \
  --output outputs/manifest.csv
```

## Train ResNet18 Baseline

```bash
PYTHONPATH=src python -m stage_age.train --config configs/resnet18.json
```

Useful quick-run override:

```bash
PYTHONPATH=src python -m stage_age.train \
  --config configs/resnet18.json \
  --epochs 2 \
  --batch-size 16 \
  --num-workers 2 \
  --output-dir outputs/debug_resnet18
```

## Outputs

Training outputs are written to timestamped run directories. With the default
config, a run directory looks like:

```text
outputs/20260511_150230_resnet18_baseline/
```

If `--output-dir outputs/debug_resnet18` is passed, the directory becomes:

```text
outputs/YYYYMMDD_HHMMSS_debug_resnet18/
```

Each training run writes:

- `config.json`: resolved config
- `manifest.csv`: image-level manifest with subject IDs, labels, and splits
- `split_summary.csv`: subject and image counts by split/class
- `train_log.csv`: epoch metrics
- `best.pt`: checkpoint selected by validation macro F1
- `test_metrics.json`: image-level and subject-level test metrics
- `test_subject_predictions.csv`: subject-level test predictions
- `figures/training_curves_latest.png`: training/validation curves, overwritten every 5 epochs and at the final epoch
- `result.md`: compact run summary with split, training, test metrics, class reports, and confusion matrices

To generate plots and `result.md` for an existing run:

```bash
PYTHONPATH=src python -m stage_age.report --run-dir outputs/resnet18_baseline
```

## Changing Age Boundaries

Change `data.bins` and `data.class_names` in `configs/resnet18.json`.

For example, a future `45/65` split can use:

```json
"bins": [18, 45, 65, 101],
"class_names": ["18-44", "45-64", "65-100"]
```
