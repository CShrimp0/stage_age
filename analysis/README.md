# Feature Profile Analysis

This folder contains subject-level ultrasound feature profiling tools for the
stage-age project.

Main entry:

```bash
PYTHONPATH=src python analysis/feature_profile_analysis.py \
  --image_dir /home/szdx/LNX/data/TA/Healthy/Images \
  --experiment_dir /home/szdx/LNX/stage-age/outputs/20260511_162523_usfm_partial_last_block_age45_65_multiseed
```

The script reads existing experiment manifests and subject-level predictions,
extracts heuristic image/ultrasound features, aggregates them to subject level,
runs exploratory statistics, and writes CSV tables, figures, and
`analysis_report.md` into a timestamped `outputs/*_feature_profile_analysis`
directory.

Current features are computed from whole valid image regions after simple
black-border removal. They are not manually annotated ROI or muscle-mask
features.

Outputs include:

- `image_features.csv`
- `subject_features.csv`
- `seed_level_subject_feature_predictions.csv`
- `subject_consensus_features.csv`
- feature significance/statistics CSVs
- heatmaps under `figures/`
- `analysis_report.md`
