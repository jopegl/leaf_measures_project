"""
Teste de ponta a ponta (precisa de torch/torchvision, roda em CPU em ~1 min):
treina 4 iterações em um dataset falso para cada modo de entrada e avalia.

    pytest tests -q
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from conftest import make_overrides  # noqa: E402
from leaf_measures.config import load_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_EXP = os.path.join(ROOT, "configs", "experiments", "resnet50_rgb.yaml")


@pytest.mark.parametrize("mode", ["rgb", "hadamard", "mask_concat"])
def test_train_resume_and_evaluate(fake_dataset, mode):
    from leaf_measures.data import folds_from_config
    from leaf_measures.evaluation.evaluator import evaluate_experiment
    from leaf_measures.training.trainer import fold_paths, train_experiment

    cfg = load_config(BASE_EXP, make_overrides(fake_dataset, **{"input.mode": mode, "experiment.name": f"t_{mode}"}))
    folds = folds_from_config(cfg)
    device = torch.device("cpu")

    train_experiment(cfg, folds, device)
    paths = fold_paths(cfg, folds[0])
    assert os.path.isfile(paths["best"]) and os.path.isfile(paths["last"])
    assert torch.load(paths["last"])["iteration"] == 4

    # rodar de novo: o fold já está completo e é pulado (retomada via last.pth)
    mtime = os.path.getmtime(paths["last"])
    cfg2 = load_config(BASE_EXP, make_overrides(fake_dataset, **{"input.mode": mode, "experiment.name": f"t_{mode}"}))
    train_experiment(cfg2, folds, device)
    assert os.path.getmtime(paths["last"]) == mtime

    # mudar train.iters num experimento existente dá erro claro
    cfg3 = load_config(BASE_EXP, make_overrides(fake_dataset, **{
        "input.mode": mode, "experiment.name": f"t_{mode}", "train.iters": 6}))
    with pytest.raises(RuntimeError, match="train.iters"):
        train_experiment(cfg3, folds, device)

    summary = evaluate_experiment(cfg2, folds, device, full=True)
    assert summary["overall"]["n_samples"] == 2
    out = os.path.join(cfg2.eval.results_dir, cfg2.experiment.name)
    for f in ("summary.json", "resumo_rer.csv", "resumo_metricas.csv"):
        assert os.path.isfile(os.path.join(out, f))
    assert os.path.isfile(os.path.join(out, "cv1_split1", "predicoes", "area_predictions.csv"))
    assert json.load(open(os.path.join(out, "summary.json")))["input_mode"] == mode


def test_signature_mismatch_is_refused(fake_dataset):
    from leaf_measures.data import folds_from_config
    from leaf_measures.training.trainer import train_experiment

    cfg = load_config(BASE_EXP, make_overrides(fake_dataset, **{"experiment.name": "t_sig"}))
    folds = folds_from_config(cfg)
    train_experiment(cfg, folds, torch.device("cpu"))
    cfg_b = load_config(BASE_EXP, make_overrides(fake_dataset, **{
        "experiment.name": "t_sig", "model.backbone": "resnet34"}))
    with pytest.raises(RuntimeError, match="outra configuração"):
        train_experiment(cfg_b, folds, torch.device("cpu"))


@pytest.mark.parametrize("backbone", ["resnet18", "efficientnet_b0", "convnext_tiny", "mobilenet_v3_small",
                                      "densenet121", "regnet_y_400mf"])
@pytest.mark.parametrize("in_ch", [3, 5])
def test_backbone_factory(backbone, in_ch):
    from leaf_measures.models.regressor import Regressor

    m = Regressor(backbone=backbone, pretrained=False, in_channels=in_ch).eval()
    with torch.no_grad():
        y = m(torch.rand(2, in_ch, 64, 64))
    assert y.shape == (2, 4)


def test_old_resnet_checkpoint_keys_compatible():
    """O state_dict do Regressor tem as mesmas chaves do antigo ResNetRegressor."""
    import torch.nn as nn
    import torchvision.models as tvm

    from leaf_measures.models.regressor import Regressor

    class OldResNetRegressor(nn.Module):  # cópia do models/resnet_regressor.py antigo
        def __init__(self):
            super().__init__()
            bb = tvm.resnet50(weights=None)
            in_f = bb.fc.in_features
            bb.fc = nn.Identity()
            self.backbone = bb
            self.head = nn.Sequential(nn.Linear(in_f, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 4))

    new = Regressor(backbone="resnet50", pretrained=False)
    new.load_state_dict(OldResNetRegressor().state_dict(), strict=True)


def test_old_resnet_mask_checkpoint_compatible():
    """O antigo ResNetMaskRegressor == Regressor com input.mode=mask_concat (5 canais, extras zerados)."""
    import torch.nn as nn
    import torchvision.models as tvm

    from leaf_measures.models.inputs import mask_concat
    from leaf_measures.models.regressor import Regressor

    class OldResNetMaskRegressor(nn.Module):  # cópia do models/resnet_mask_regressor.py antigo
        def __init__(self):
            super().__init__()
            bb = tvm.resnet50(weights=None)
            c = bb.conv1
            new = nn.Conv2d(5, c.out_channels, kernel_size=c.kernel_size, stride=c.stride,
                            padding=c.padding, bias=c.bias is not None)
            with torch.no_grad():
                new.weight[:, :3] = c.weight
                new.weight[:, 3:] = 0
            bb.conv1 = new
            in_f = bb.fc.in_features
            bb.fc = nn.Identity()
            self.backbone = bb
            self.head = nn.Sequential(nn.Linear(in_f, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, 4))

        def forward(self, x, masks=None):
            if masks is not None:
                x = torch.cat([x, masks], dim=1)
            return self.head(self.backbone(x))

    old = OldResNetMaskRegressor().eval()
    new = Regressor(backbone="resnet50", pretrained=False, in_channels=5).eval()
    new.load_state_dict(old.state_dict(), strict=True)

    image, masks = torch.rand(2, 3, 64, 64), (torch.rand(2, 2, 64, 64) > 0.5).float()
    with torch.no_grad():
        assert torch.allclose(old(image, masks), new(mask_concat(image, masks)), atol=1e-5)


@pytest.mark.parametrize("init", ["zeros", "mean"])
def test_extra_channels_init(init):
    from leaf_measures.models.regressor import Regressor

    w = Regressor(backbone="resnet18", pretrained=False, in_channels=5, extra_channels_init=init).backbone.conv1.weight
    if init == "zeros":
        assert torch.count_nonzero(w[:, 3:]) == 0
    else:
        assert torch.allclose(w[:, 3:], w[:, :3].mean(dim=1, keepdim=True).expand(-1, 2, -1, -1))
