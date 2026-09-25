"""Part 4: optimizer and scheduler wiring."""

import torch

from seesense_lab.model import SmallResNet
from seesense_lab.train import build_optimizer_and_scheduler


def test_optimizer_settings():
    opt, _ = build_optimizer_and_scheduler(SmallResNet(), lr=0.05, epochs=10)
    g = opt.param_groups[0]
    assert isinstance(opt, torch.optim.SGD)
    assert g["lr"] == 0.05 and g["momentum"] == 0.9 and g["nesterov"] is True
    assert g["weight_decay"] == 5e-4


def test_cosine_schedule_shape():
    opt, sched = build_optimizer_and_scheduler(SmallResNet(), lr=0.1, epochs=10)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == 0.1
    assert 0.045 < lrs[5] < 0.055          # halfway through: about half of max
    assert lrs[-1] < 0.01                   # ends near zero
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_scheduler_can_be_disabled():
    _, sched = build_optimizer_and_scheduler(SmallResNet(), use_scheduler=False)
    assert sched is None
