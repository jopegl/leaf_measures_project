"""Perdas: as antigas não mudaram e as *_log medem erro relativo."""
import pytest

torch = pytest.importorskip("torch")

from leaf_measures.data.normalization import MEASURE_ORDER, normalize  # noqa: E402
from leaf_measures.training.losses import weighted_measures_loss  # noqa: E402

W = torch.ones(4)


def norm(values):
    """values: lista de linhas [length, width, perimeter, area] em cm/cm² -> tensor normalizado."""
    t = torch.tensor(values, dtype=torch.float32)
    return torch.stack([normalize(m, t[:, i]) for i, m in enumerate(MEASURE_ORDER)], dim=1)


def test_huber_unchanged():
    pred, target = torch.rand(8, 4), torch.rand(8, 4)
    d = (pred - target).abs()
    ref = torch.where(d < 0.1, 0.5 * d ** 2 / 0.1, d - 0.05).mean(0)
    loss, parts = weighted_measures_loss(pred, target, W, "huber", 0.1)
    assert torch.allclose(parts, ref) and torch.isclose(loss, ref.sum())


def test_log_loss_zero_when_exact():
    t = norm([[5.0, 7.0, 25.0, 25.0]])
    for kind in ("mse_log", "huber_log", "l1_log"):
        assert weighted_measures_loss(t, t, W, kind, 0.05)[0].item() == pytest.approx(0.0, abs=1e-6)


def test_log_loss_is_relative():
    """O mesmo erro absoluto de área (1,5 cm²) custa mais na folha pequena."""
    small_t, small_p = norm([[5, 7, 25, 15.0]]), norm([[5, 7, 25, 16.5]])
    big_t, big_p = norm([[5, 7, 25, 40.0]]), norm([[5, 7, 25, 41.5]])
    for kind in ("mse_log", "huber_log", "l1_log"):
        small = weighted_measures_loss(small_p, small_t, W, kind, 0.05)[1][3]
        big = weighted_measures_loss(big_p, big_t, W, kind, 0.05)[1][3]
        assert small > 2 * big, kind
    # na Huber antiga (normalizada) os dois custam o mesmo
    s = weighted_measures_loss(small_p, small_t, W, "huber", 0.1)[1][3]
    b = weighted_measures_loss(big_p, big_t, W, "huber", 0.1)[1][3]
    assert s.item() == pytest.approx(b.item(), rel=1e-4)


def test_log_loss_symmetric_over_under():
    """Errar 10% para cima ou 10% para baixo (em razão) custa igual — sem viés pró-subestimação."""
    t = norm([[5, 7, 25, 25.0]])
    over, under = norm([[5, 7, 25, 25.0 * 1.1]]), norm([[5, 7, 25, 25.0 / 1.1]])
    for kind in ("mse_log", "l1_log"):
        lo = weighted_measures_loss(over, t, W, kind)[1][3]
        lu = weighted_measures_loss(under, t, W, kind)[1][3]
        assert lo.item() == pytest.approx(lu.item(), rel=1e-4)


def test_log_loss_survives_nonpositive_predictions():
    pred = torch.full((2, 4), -5.0, requires_grad=True)   # bem abaixo de zero em unidades reais
    loss, _ = weighted_measures_loss(pred, norm([[5, 7, 25, 25.0], [6, 8, 28, 30.0]]), W, "huber_log", 0.05)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()


def test_unknown_type():
    with pytest.raises(ValueError):
        weighted_measures_loss(torch.zeros(1, 4), torch.zeros(1, 4), W, "huber_log2")
