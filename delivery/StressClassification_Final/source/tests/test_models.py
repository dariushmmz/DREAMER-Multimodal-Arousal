import torch

from stress_classification.models import create_model, parameter_counts


def test_models_produce_three_logits():
    eeg = torch.zeros(2, 1280, 14)
    ecg = torch.zeros(2, 2560, 1)
    hrv = torch.zeros(2, 25)
    for name in ("base_cnn", "paper_inspired_attention"):
        model = create_model(name)
        model.eval()
        with torch.no_grad():
            output = model(eeg, ecg, hrv)
        assert output.shape == (2, 3)
        counts = parameter_counts(model)
        assert counts["total"] == counts["trainable"]
        assert counts["total"] > 0

