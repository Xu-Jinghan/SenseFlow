# SenseFlow

SenseFlow is a measurement-driven, cross-layer framework for evaluating emerging optoelectronic sensors in intelligent vision systems. It models dynamic photodetector non-idealities, sensor-array readout, CIFAR/KITTI task inference, CIM functional non-idealities, and NeuroSim hardware metrics in one workflow.

This repository accompanies **"SenseFlow: A Unified Framework for Evaluating Emerging Optoelectronic Devices in Intelligent Vision Systems"** (DAC 2026, DOI: `10.1145/3770743.3804102`).

## Contents

- `photodetector_model.py`: unified dynamic photodetector model, state updates, current generation, and temporal-noise synthesis.
- `photodetector_array.py`: array-level simulation and readout models.
- `fit_*.py`: extraction of steady-state, transient, drift, and noise parameters from measurements.
- `data/`: small measurement and literature-summary CSV files used by fitting and plotting scripts.
- `NeuroSim-2DInferenceV1.5/NeuroSim-2DInferenceV1.5/`: CIFAR/KITTI pipelines, restoration, DSE, CIM functional simulation, and NeuroSim integration.
- `paper/`: published figure panels and a figure-to-code reproduction index.

Datasets, checkpoints, generated images, logs, cached tensors, and compiled objects are intentionally not included.

## Installation

CUDA is strongly recommended for full CIFAR/KITTI and CIM evaluations. Sensor-model smoke tests can run on CPU.

```bash
conda env create -f environment.yml
conda activate senseflow
python -m pip install -e NeuroSim-2DInferenceV1.5/NeuroSim-2DInferenceV1.5/pytorch-quantization
python -c "import torch; print(torch.cuda.is_available())"
```

Place datasets under `.datasets/`. Put checkpoints under `NeuroSim-2DInferenceV1.5/NeuroSim-2DInferenceV1.5/models/`; checkpoint files are ignored by Git.

## Entry points

Sensor model:

```bash
python simulate_random_square_from_params.py --help
python photodetector_model.py --help
```

CIFAR and sensor/CIM evaluation:

```bash
cd NeuroSim-2DInferenceV1.5/NeuroSim-2DInferenceV1.5
python inference.py --help
python eval_resnet18_sensor_video_sequence.py --help
```

Use `--hardware 0` to isolate sensor/model evaluation and `--hardware 1` to enable CIM functional simulation. NeuroSim PPA evaluation additionally requires compiling `NeuroSIM/`.

KITTI evaluation and restoration:

```bash
python eval_yolo11n_kitti_tracking_sensor.py --help
python train_kitti_restoration_yolo_sensor.py --help
```

KITTI data and Ultralytics weights must be obtained separately under their own terms.

## Paper reproduction

See [`paper/README.md`](paper/README.md) for the figure-by-figure mapping. Published panels are retained as visual references; scripts regenerate outputs into ignored directories.

## Licensing

Original SenseFlow code is released under Apache-2.0. Bundled third-party components retain their own licenses. NeuroSim is distributed under CC BY-NC 4.0 and restricted to non-commercial use; see [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Citation

```bibtex
@inproceedings{xu2026senseflow,
  title     = {SenseFlow: A Unified Framework for Evaluating Emerging Optoelectronic Devices in Intelligent Vision Systems},
  author    = {Xu, Jinghan and Zhang, Ligong and Li, Jiaqi and Wang, Shuhan and Liu, Yizhan and Zhu, Haotong and Qin, Xionghao and Huang, Peng and Zhou, Zheng and Liu, Xiaoyan},
  booktitle = {Proceedings of the 63rd ACM/IEEE Design Automation Conference},
  year      = {2026},
  doi       = {10.1145/3770743.3804102}
}
```
