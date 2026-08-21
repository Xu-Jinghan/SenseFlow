# Paper figure reproduction

`published/` contains the raster panels embedded in the camera-ready manuscript and serves as a visual reference. `image14.png` and `image15.png` are the two panels of Figure 14. Published files are not algorithm inputs.

| Figure | Topic | Main implementation / reproduction entry point |
| --- | --- | --- |
| 1 | Conventional and emerging pipelines | Conceptual artwork; published reference only |
| 2 | SenseFlow architecture | `photodetector_model.py`, `photodetector_array.py`, `sensor_video_sequence_backend.py`, `inference.py` |
| 3 | Parameter extraction | Original extraction flow in `photodetector_model.py`; reusable CSV/image inputs are supported by `fit_waveform_from_csv.py` and `fit_waveform_from_image.py` |
| 4 | Unified discrete-ODE response | `photodetector_model.py`, `simulate_random_square_from_params.py` |
| 5 | Temporal sequence simulation | `generate_sensor_verification_images_video_sequence.py`, `sensor_video_sequence_backend.py` |
| 6 | Delay and on-state drift | `evaluate_case1_baseline_vs_eta0p2_drift_video_sequence.py` |
| 7 | Nonlinearity and aging | `photodetector_model.py`, `generate_sensor_verification_images_video_sequence.py` |
| 8 | Noise and spatial mismatch | `evaluate_case1_native_noise_scale_video_sequence.py`, `evaluate_case1_native_spatial_variation_video_sequence.py` |
| 9 | Joint sensor/CIM impact | `inference.py`, `plot_sensor_cim_spatial_variation_sweep.py` |
| 10 | Literature device metrics | `analyze_photodetector_paper_dataset.py` and `data/photodetector_paper_dataset.csv` |
| 11 | Accuracy design maps | `scan_cifar10_fixed_eta_noise_full_log_grid.py`, `plot_cifar10_fixed_eta_r_tr_noise_paper.py` |
| 12 | Feasible regions/device overlay | `plot_cifar10_contour_with_all_structures_overlay.py`, `plot_dual_scan_overview_with_paper_structures.py` |
| 13 | Multi-frame restoration | `models/restoration.py`, `train_restoration_resnet18_sensor_video_sequence.py` |
| 14 | Restoration results | `eval_restoration_resnet18_sensor_video_sequence.py`, `plot_restoration_recovery_overview.py`, `export_restoration_visuals_video_sequence.py` |

Unprefixed pipeline paths are relative to `NeuroSim-2DInferenceV1.5/NeuroSim-2DInferenceV1.5/`. Most task plots require evaluation JSON/CSV files generated first by the corresponding `eval_*`, `scan_*`, or `sweep_*` script. Run each entry point with `--help` to inspect dataset, checkpoint, seed, FPS, readout, and output options.

## Verified commands

The following commands were executed successfully from a fresh clone on Windows with the `xjhenv` conda environment. Replace `conda run -n xjhenv` with your Python environment when appropriate.

Case 1 model and ODE figures:

```bash
conda run -n xjhenv python photodetector_model.py --case case1
```

Case 2 parameter and noise figures:

```bash
conda run -n xjhenv python photodetector_model.py \
  --case case2 \
  --case2-output-dir validation_outputs/case2_model
```

Literature-device statistics used by Figure 10:

```bash
conda run -n xjhenv python analyze_photodetector_paper_dataset.py \
  --input data/photodetector_paper_dataset.csv \
  --output-dir validation_outputs/figure10
```

Minimal CIFAR sensor-image generation without a checkpoint:

```bash
cd NeuroSim-2DInferenceV1.5/NeuroSim-2DInferenceV1.5
conda run -n xjhenv python generate_sensor_verification_images_video_sequence.py \
  --data-root <DATA_ROOT> \
  --source-dataset cifar10 \
  --split test \
  --generate-images 1 \
  --run-eval 0 \
  --num-images 2 \
  --range-calibration-samples 8 \
  --fps-sim 100 \
  --use-noise-fn 0 \
  --output-dir validation_outputs/figure5_smoke
```

Minimal CIFAR `raw/ideal/nonideal` inference requires a trained ResNet18 checkpoint:

```bash
conda run -n xjhenv python eval_resnet18_sensor_video_sequence.py \
  --data_path <DATA_ROOT> \
  --source_dataset cifar10 \
  --model_path <RESNET18_CIFAR10_CHECKPOINT> \
  --cases raw ideal nonideal \
  --batch_size 2 \
  --max_eval_batches 1 \
  --generate_images 1 \
  --num_images 2 \
  --range_calibration_samples 8 \
  --fps_sim 100 \
  --use_noise_fn 0 \
  --output_dir validation_outputs/cifar_eval_smoke \
  --results_json validation_outputs/cifar_eval_smoke/results.json
```

Minimal KITTI `clean/ideal/nonideal` detection requires KITTI Tracking data and a YOLO11n checkpoint:

```bash
conda run -n xjhenv python eval_yolo11n_kitti_tracking_sensor.py \
  --kitti-root <KITTI_TRACKING_ROOT> \
  --model <YOLO11N_CHECKPOINT> \
  --sequence 0000 \
  --start-frame 0 \
  --num-frames 1 \
  --eval-cases clean ideal nonideal \
  --output-width 320 \
  --output-height 96 \
  --imgsz 320 \
  --batch-size 1 \
  --device 0 \
  --output-dir validation_outputs/kitti_smoke
```

CIFAR datasets can normally be downloaded by torchvision. KITTI data, ResNet checkpoints, YOLO checkpoints, and restoration checkpoints are external assets and are intentionally excluded from Git.
