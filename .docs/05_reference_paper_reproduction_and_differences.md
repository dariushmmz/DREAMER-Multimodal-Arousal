# Comparison with the 2026 reference paper

## Reference identity

The sole client-requested reference is the 2026 *Scientific Reports* article
[*An attention-based multimodal deep learning framework integrating EEG and
ECG for enhanced stress detection*](https://doi.org/10.1038/s41598-026-44499-0).
The local source is `.paper/An attention-based multimodal.pdf`.

The separate stacking ensemble retained in the seven-model comparison is a
classical comparator. It is not presented as a reproduction or baseline from
the 2026 reference paper.

## Method reported by the paper

The PDF reports 10-second non-overlapping epochs; EEG scalograms processed in
parallel by modified VGG16, EfficientNetB0, and ResNeXt50 networks; a raw-ECG
CNN plus 25 HRV features; hierarchical attention; a 512→256→128→3
classification head; Adam/cosine training; a subject-independent 80/10/10
division; approximately 42.7 million parameters; and 95.7% accuracy for the
three states rest, stress, and amusement.

## Reproducibility boundary

A faithful reproduction could not be established from the available evidence:

- The official [WESAD description](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
  lists BVP, ECG, EDA, EMG, respiration, temperature, and acceleration for 15
  participants; it does not document the EEG configuration attributed to WESAD
  in the paper.
- The primary [CASE data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC6785543/)
  lists ECG, BVP, three EMG channels, GSR, respiration, and skin temperature for
  30 participants; it likewise does not document the EEG configuration
  attributed to CASE in the paper.
- The paper describes a 35-participant population, 700 Hz EEG/ECG inputs, and a
  28/4/3 participant split that cannot be reconstructed from the documented
  public WESAD and CASE releases alone.
- No executable implementation or complete data-to-label mapping was available
  locally to resolve those differences.

These observations identify an evidence gap; they do not establish that no
private or transformed data existed. They do mean that claiming a faithful
reproduction from the public evidence available to this project would be
scientifically unsupported.

## Main differences

| Dimension | 2026 paper report | Corrected project |
|---|---|---|
| Dataset | WESAD + CASE, reported 35 participants | DREAMER, 23 participants |
| Target | Rest/stress/amusement protocol states | Self-reported low/medium/high arousal |
| Modalities | Reported EEG + ECG + HRV | DREAMER 14-channel EEG + raw ECG + ECG-derived HRV |
| Epochs | 10 s, no overlap | 10 s, no overlap |
| Split | Reported subject-independent 80/10/10 | Fixed subject-independent 17/3/3 |
| EEG representation | CWT 128×128×3, three image backbones | Raw temporal multi-scale Conv1D adaptation |
| ECG representation | Raw CNN then image backbones + 25 HRV features | Raw Conv1D + 25 HRV descriptors |
| Fusion | Hierarchical attention | Learned two-way modality attention |
| Parameters | Approximately 42.7 M | 1.846 M |
| Scientific claim | Stress-state detection | Arousal classification only |

## Adapted concepts and omitted components

The implementation named `paper_inspired_attention` adapts the paper's
non-overlapping 10-second segmentation, multi-scale EEG processing, joint
raw-ECG and HRV representation, attention-based modality fusion, a
512→256→128→3 head, Adam-family optimization, and cosine scheduling.

It does not reproduce the WESAD+CASE population, rest/stress/amusement labels,
CWT images, modified VGG16/EfficientNetB0/ResNeXt50 branches, hierarchical
image-feature attention, 42.7-million-parameter scale, or the reported split.
These are substantive architectural and experimental differences.

## Empirical outcome of the DREAMER adaptation

Under the corrected DREAMER protocol, `paper_inspired_attention` selected
epoch 12 by validation macro-F1 and achieved test accuracy 0.4029, balanced
accuracy 0.3732, macro-F1 0.3554, and macro-AUROC 0.5117. Base CNN achieved
0.3297, 0.3584, 0.2394, and 0.4843 respectively. The adaptation therefore
improves macro-F1 over Base CNN by 0.1160.

The separately evaluated classical stacking comparator obtained macro-F1
0.3755. It remains in the general seven-model comparison but is not attributed
to the 2026 paper and is not evidence of reproducing that paper.

## Interpretation

The DREAMER measurement neither reproduces nor refutes the paper's reported
95.7% accuracy. The datasets, labels, sensor descriptions, representations,
parameter counts, and evaluation populations are not equivalent. The supported
conclusion is limited to this project: the adapted attention concepts improve
over the implemented Base CNN, while cross-participant DREAMER arousal
performance remains modest and variable.
