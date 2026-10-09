# Parkinson's Disease Detection from Gait Time Series with Mamba

**Deep learning for Parkinson's disease classification using ground-reaction-force signals**

This repository investigates the use of modern sequence models for Parkinson's disease (PD) detection from gait recordings, with a particular focus on **Mamba state-space models** and their comparison with recurrent neural networks and Transformers.

The project uses the [Gait in Parkinson's Disease dataset](https://physionet.org/content/gaitpdb/1.0.0/) from PhysioNet. It explores how temporal modeling architectures learn gait dynamics for binary classification of Parkinson's disease versus healthy controls, with additional experiments targeting disease severity.

## Overview

Parkinson's disease affects motor function and can alter gait patterns. Ground-reaction-force (GRF) measurements provide a time-resolved representation of these changes, making gait analysis a useful setting for investigating time-series classification methods.

This project evaluates whether state-space sequence models can effectively capture temporal patterns in gait signals and how their performance compares with established deep learning architectures.

The main objectives are to:

* **Evaluate Mamba for biomedical time-series classification**, using multichannel gait recordings.
* **Compare sequence architectures**, including Mamba, LSTM, GRU, and Transformer-based models.
* **Investigate temporal context**, by evaluating different input window lengths.
* **Prevent data leakage**, using subject-level separation between training and validation data.
* **Explore clinical classification tasks**, including PD-versus-control classification and severity classification within the PD cohort.

## Dataset

The experiments use the [PhysioNet Gait in Parkinson's Disease dataset](https://physionet.org/content/gaitpdb/1.0.0/).

The dataset contains recordings from **166 subjects: 93 people with Parkinson's disease and 73 healthy controls**. The recordings contain vertical ground-reaction-force measurements collected from sensors beneath the feet.

### Signal representation

The original recordings include:

* Time measurements.
* Eight vertical ground-reaction-force (VGRF) channels for the left foot.
* Eight VGRF channels for the right foot.
* Left- and right-foot aggregate force measurements.

The modeling pipeline uses an **18-channel representation** of the gait signals, retaining information from both feet and the aggregate measurements.

The recordings are sequential rather than independent observations. Consequently, multiple files or windows from the same participant must not be treated as independent subjects during evaluation.

### Classification tasks

**1. Binary classification**

Distinguish between:

* Parkinson's disease (PD)
* Healthy controls

**2. Disease severity classification**

Investigate classification within the Parkinson's disease cohort using Hoehn and Yahr (H&Y) severity labels, including the classes represented in the available recordings.

Severity classification is a distinct task from binary diagnosis and should be evaluated on the PD cohort rather than interpreted as a three-class version of the binary task.

## Models

The repository organizes experiments around Mamba and alternative sequence architectures.

| Architecture | Role                                                               |
| ------------ | ------------------------------------------------------------------ |
| **Mamba**    | Selective state-space sequence modeling for temporal gait patterns |
| LSTM         | Recurrent baseline for sequential dependencies                     |
| GRU          | Recurrent baseline with gated state updates                        |
| Transformer  | Attention-based modeling of temporal relationships                 |

Mamba uses input-dependent state-space dynamics to process sequences. Its ability to model long temporal contexts makes it an interesting candidate for gait recordings, where clinically relevant patterns may extend beyond short local windows.

The recurrent and attention-based baselines provide reference points for evaluating whether the state-space approach offers practical benefits on this dataset.

## Experimental methodology

### Temporal windowing

The experiments investigate multiple sequence lengths, including windows of 100, 400, 800, 1,200, and 2,000 time steps.

Varying the window length helps assess the trade-off between local gait information and longer temporal context. Longer sequences may expose broader gait dynamics, but also increase computational requirements and optimization difficulty.

### Subject-level data separation

A central methodological requirement is to prevent subject leakage.

Recordings are grouped by participant when constructing training and validation splits. This prevents recordings from the same individual from appearing in both sets and artificially inflating performance through participant-specific patterns.

This is particularly important when a dataset contains multiple recordings per subject.

### Evaluation

Model performance is assessed using classification metrics, including:

* Accuracy
* Macro-F1 score
* Class-specific precision and recall

Macro-F1 gives equal weight to each class, making it useful when class frequencies differ. Class-specific recall also helps identify whether a model disproportionately misses Parkinson's disease cases or healthy controls.

Results should be compared under consistent subject-level splits and equivalent preprocessing protocols.

## Results

In a leakage-controlled binary classification experiment, the following results were obtained:

| Model     |   Accuracy |   Macro-F1 |
| --------- | ---------: | ---------: |
| **Mamba** | **88.61%** | **88.52%** |
| LSTM      |     87.18% |     87.16% |

For the reported Mamba experiment, PD recall was 80.20%, while control recall was 96.97%.

These results indicate that Mamba outperformed the LSTM baseline in this particular experiment. The difference is encouraging, but does not by itself establish statistical significance or prove that Mamba is superior across all architectures, sequence lengths, or data splits.

The broader objective is to characterize when state-space models are effective for gait classification and how their performance changes with temporal context and model architecture.

## Repository structure

The repository is organized into experiment-specific directories:

```text
PD_Detection_Mamba/
├── classification/          # Classification-related experiments
├── data/
│   └── processed/           # Processed data
├── lstm_gru_transformer/     # Recurrent and Transformer experiments
├── mamba/                   # Mamba experiments
├── results_mamba/           # Mamba experiment results
├── testing/                 # Testing and experimental utilities
├── Alex_Martinelli_Report.pdf
├── Martinelli_FAU_Gait_Mamba.pptx
├── handcrafted_features_description.txt
└── svm_results.txt
```

See the individual experiment directories for the corresponding implementation, configuration, and evaluation scripts.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/AlexM3020/PD_Detection_Mamba.git
cd PD_Detection_Mamba
```

### 2. Prepare the environment

The experiments use Python and PyTorch-based deep learning workflows. Mamba experiments additionally require a compatible Mamba-SSM installation and, for the accelerated implementation, a compatible PyTorch, CUDA, compiler, and GPU environment.

Create a dedicated Python environment before installing dependencies. Check the scripts and configuration files in the relevant experiment directory for the required package versions and dataset paths.

### 3. Prepare the data

Download the Gait in Parkinson's Disease dataset from [PhysioNet](https://physionet.org/content/gaitpdb/1.0.0/) and prepare the processed recordings according to the preprocessing pipeline used by the selected experiment.

Update any dataset paths in the relevant scripts or configuration files before running an experiment.

### 4. Run an experiment

Choose the appropriate experiment directory (`mamba/` or `lstm_gru_transformer/`) and follow the entry point and configuration used for that experiment.

For a fair comparison, keep the subject-level data split, preprocessing, input window length, and evaluation metrics consistent across architectures.

## Limitations

* The dataset is relatively small compared with large-scale clinical datasets.
* Performance on this dataset does not establish generalization to unseen hospitals, sensor systems, or populations.
* Long input sequences can increase memory use and make training more difficult.
* Binary classification performance does not directly establish the reliability of disease-severity classification.
* The reported results are experimental findings, not evidence of clinical readiness.

This repository is a research project and **is not intended for clinical diagnosis or medical decision-making**.

## References

1. PhysioNet. *Gait in Parkinson's Disease*. https://physionet.org/content/gaitpdb/1.0.0/
2. Gu, A. and Dao, T. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752. https://arxiv.org/abs/2312.00752

## Author

**Alex Martinelli**

M.Sc. Data Science, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)

For further details on the experimental setup and research findings, see the report and presentation included in this repository.
