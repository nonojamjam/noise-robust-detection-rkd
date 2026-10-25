"""Synthetic smoke test for probe_run's streaming ridge and correspondence path."""
import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location("probe_run", Path(__file__).with_name("probe_run.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def normalize(x):
    x = x.permute(0, 2, 3, 1)
    return torch.nn.functional.normalize(torch.nn.functional.layer_norm(x, (x.shape[-1],), eps=1e-6), dim=-1, eps=1e-6).permute(0, 3, 1, 2)


def main():
    test_center_context()
    test_neighborhood_context()


def test_center_context():
    g = torch.Generator().manual_seed(7)
    s = torch.randn((2, 256, 32, 32), generator=g)
    mask = torch.ones((2, 32, 32), dtype=torch.bool)
    # The target is exactly a linear map of the standardised student features.
    mean, std = s.mean((0, 2, 3)), s.std((0, 2, 3), unbiased=False).clamp_min(1e-6)
    x = probe.design_rows(s, mask, mean, std)
    true_w = torch.randn((257, 768), generator=g) / 16
    raw = torch.zeros((2, 32, 32, 768)); raw[mask] = x @ true_w
    target = normalize(raw.permute(0, 3, 1, 2))
    eq = probe.NormalEquations(); eq.add(x, target.permute(0, 2, 3, 1)[mask])
    scale = float(torch.trace(eq.xtx) / 257)
    losses = []
    for factor in probe.LAM_FACTORS:
        w = probe.ridge_weights(eq, factor * scale)
        pred = probe.predict(s, mask, mean, std, w, normalize)
        losses.append(float((1 - (pred * target).sum(1)[mask]).mean()))
    best = min(losses)
    w = probe.ridge_weights(eq, probe.LAM_FACTORS[losses.index(best)] * scale)
    pred = probe.predict(s, mask, mean, std, w, normalize)
    # Use the exact, reused metric implementation on its full 32x32-cell path.
    api = probe.load_spatial_api(Path(__file__).resolve().parents[1] / "code_v2" / "seed_sweep_v2.py")
    stats = api["correspondence_stats"](pred, target, mask)
    assert best < 3e-4, best
    assert stats["p_at_1"] > .995, stats
    print({"test": "center_context", "best_holdout_style_loss": best, **stats})


def context_loss(s_fit, target_fit, s_hold, target_hold, context):
    mask_fit = torch.ones(s_fit.shape[0:1] + s_fit.shape[2:], dtype=torch.bool)
    mask_hold = torch.ones(s_hold.shape[0:1] + s_hold.shape[2:], dtype=torch.bool)
    moments = probe.FeatureMoments(256)  # 통계는 256 채널 기준 (probe rev2)
    moments.add(s_fit, mask_fit, context)
    mean, std = moments.finish()
    eq = probe.NormalEquations(inputs=256 * context * context + 1, outputs=target_fit.shape[1])
    eq.add(probe.design_rows(s_fit, mask_fit, mean, std, context),
           target_fit.permute(0, 2, 3, 1)[mask_fit])
    lam = 1e-3 * float(torch.trace(eq.xtx) / eq.xtx.shape[0])
    weights = probe.ridge_weights(eq, lam)
    pred = probe.predict(s_hold, mask_hold, mean, std, weights, normalize, context)
    return float((1 - (pred * target_hold).sum(1)[mask_hold]).mean())


def test_neighborhood_context():
    g = torch.Generator().manual_seed(19)
    s_fit = torch.randn((3, 256, 32, 32), generator=g)
    s_hold = torch.randn((2, 256, 32, 32), generator=g)
    # Each teacher channel is a different off-center student-neighborhood value.
    def targets(s):
        patches = probe.contextual_features(s, 3)
        raw = torch.stack([patches[..., i * 9 + j]
                           for i, j in enumerate((0, 1, 2, 3, 5, 6, 7, 8))], dim=-1)
        return normalize(raw.permute(0, 3, 1, 2))
    target_fit, target_hold = targets(s_fit), targets(s_hold)
    loss_1 = context_loss(s_fit, target_fit, s_hold, target_hold, context=1)
    loss_3 = context_loss(s_fit, target_fit, s_hold, target_hold, context=3)
    assert loss_3 < loss_1, (loss_1, loss_3)
    print({"test": "neighborhood_context", "context_1_holdout_loss": loss_1,
           "context_3_holdout_loss": loss_3})


if __name__ == "__main__":
    main()
