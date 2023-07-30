import torch

from mmpose.models import build_loss


def test_smooth_l1_loss():
    # test SmoothL1Loss without target weight
    loss_cfg = dict(type='SmoothL1Loss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(.5))

    # test SmoothL1Loss with target weight
    loss_cfg = dict(type='SmoothL1Loss', use_target_weight=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(.5))


def test_wing_loss():
    # test WingLoss without target weight
    loss_cfg = dict(type='WingLoss')
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.gt(loss(fake_pred, fake_label, None), torch.tensor(.5))

    # test WingLoss with target weight
    loss_cfg = dict(type='WingLoss', use_target_weight=True)
    loss = build_loss(loss_cfg)

    fake_pred = torch.zeros((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 2))
    fake_label = torch.zeros((1, 3, 2))
    assert torch.gt(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(.5))


def test_mse_regression_loss():

    # w/o target weight
    loss_cfg = dict(type='MSELoss')
    loss = build_loss(loss_cfg)
    fake_pred = torch.zeros((1, 3, 3))
    fake_label = torch.zeros((1, 3, 3))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 3))
    fake_label = torch.zeros((1, 3, 3))
    assert torch.allclose(loss(fake_pred, fake_label, None), torch.tensor(1.))

    # w/ target weight
    loss_cfg = dict(type='MSELoss', use_target_weight=True)
    loss = build_loss(loss_cfg)
    fake_pred = torch.zeros((1, 3, 3))
    fake_label = torch.zeros((1, 3, 3))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(0.))

    fake_pred = torch.ones((1, 3, 3))
    fake_label = torch.zeros((1, 3, 3))
    assert torch.allclose(
        loss(fake_pred, fake_label, torch.ones_like(fake_label)),
        torch.tensor(1.))
