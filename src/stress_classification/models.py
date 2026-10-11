"""Corrected deep baselines for raw EEG/ECG plus ECG-derived HRV features."""

from __future__ import annotations

import torch
from torch import nn


class TemporalBranch(nn.Module):
    def __init__(self, in_channels: int, channels: tuple[int, ...], kernels: tuple[int, ...]) -> None:
        super().__init__()
        if len(channels) != len(kernels):
            raise ValueError("channels and kernels must have equal lengths")
        layers: list[nn.Module] = []
        current = in_channels
        for output, kernel in zip(channels, kernels):
            layers.extend(
                [
                    nn.Conv1d(current, output, kernel_size=kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(output),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                ]
            )
            current = output
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.features = nn.Sequential(*layers)
        self.output_dim = channels[-1]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.features(values.transpose(1, 2)).squeeze(-1)


class BaseMultimodalCNN(nn.Module):
    """Compact early-feature-fusion baseline."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.eeg = TemporalBranch(14, (32, 64, 128), (7, 5, 3))
        self.ecg = TemporalBranch(1, (32, 64, 128), (15, 9, 5))
        self.hrv = nn.Sequential(nn.Linear(25, 64), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, eeg: torch.Tensor, ecg: torch.Tensor, hrv: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([self.eeg(eeg), self.ecg(ecg), self.hrv(hrv)], dim=1))


class MultiScaleEEG(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                TemporalBranch(14, (32, 64, 128), kernels)
                for kernels in ((15, 7, 3), (31, 15, 7), (63, 31, 15))
            ]
        )
        self.projection = nn.Sequential(nn.Linear(128 * 3, 512), nn.ReLU(inplace=True))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([branch(eeg) for branch in self.branches], dim=1))


class ECGHRVPathway(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw = TemporalBranch(1, (64, 128, 256), (15, 7, 3))
        self.hrv = nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True))

    def forward(self, ecg: torch.Tensor, hrv: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([self.raw(ecg), self.hrv(hrv)], dim=1))


class ModalityAttention(nn.Module):
    """Learn a normalized EEG/ECG contribution for each sample."""

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()
        self.eeg_value = nn.Linear(feature_dim, feature_dim)
        self.ecg_value = nn.Linear(feature_dim, feature_dim)
        self.eeg_score = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.ecg_score = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, eeg: torch.Tensor, ecg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(torch.cat([self.eeg_score(eeg), self.ecg_score(ecg)], dim=1), dim=1)
        fused = weights[:, :1] * self.eeg_value(eeg) + weights[:, 1:] * self.ecg_value(ecg)
        return fused, weights


class PaperInspiredAttentionNet(nn.Module):
    """DREAMER-compatible adaptation of the 2026 paper's fusion concept.

    This is deliberately named *inspired*: it does not claim to reproduce the
    paper's CWT/ImageNet triple-backbone system, which cannot be reconstructed
    from the paper's stated WESAD/CASE inputs because those datasets lack EEG.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.eeg = MultiScaleEEG()
        self.ecg = ECGHRVPathway()
        self.attention = ModalityAttention(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.last_attention_weights: torch.Tensor | None = None

    def forward(self, eeg: torch.Tensor, ecg: torch.Tensor, hrv: torch.Tensor) -> torch.Tensor:
        fused, weights = self.attention(self.eeg(eeg), self.ecg(ecg, hrv))
        self.last_attention_weights = weights.detach()
        return self.classifier(fused)


def create_model(name: str, num_classes: int = 3) -> nn.Module:
    normalized = name.lower().strip()
    if normalized == "base_cnn":
        return BaseMultimodalCNN(num_classes=num_classes)
    if normalized == "paper_inspired_attention":
        return PaperInspiredAttentionNet(num_classes=num_classes)
    raise KeyError(f"unknown model {name!r}")


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }

