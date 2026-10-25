"""SPEC v2.1 §8 / SPEC_A §11 CPU regression and integration tests.

Run from workspace with the specified interpreter; all temporary files stay here.
The real cached DINOv2 is used once if available; no downloads are permitted.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
# Keep library caches and subprocess output inside the authorized workspace.
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'tests' / '.runtime' / 'ultralytics')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'tests' / '.runtime' / 'matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
# Ultralytics falls back to /tmp if this parent does not already exist.
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import C2f
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.models.yolo.segment import SegmentationTrainer


torch.set_num_threads(2)


def test_library_config_stays_in_workspace():
    from ultralytics.utils import USER_CONFIG_DIR
    assert USER_CONFIG_DIR.is_relative_to(ROOT)


def load_module(exp='S1', path='seed_sweep_v2.py', **settings):
    name = f'_spatial_test_{Path(path).stem}_{exp}_{len(sys.modules)}'
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    module.EXP = exp
    module.__SRC_PATH__ = str(ROOT / path)
    module.__dict__.update(settings)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def kd():
    return load_module()


class FakeViT(nn.Module):
    """Small deterministic ViT-shaped fixture: raw tokens are deliberately not LN'd."""
    def __init__(self, bad_shape=False, fail=False):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.weight = nn.Parameter(torch.ones(()))
        self.bad_shape = bad_shape
        self.fail = fail
        self.seen = []
        self.batch_sizes = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        self.batch_sizes.append(x.shape[0])
        # Depend on the image, channel and position, including normalization-sensitive norms.
        pooled = F.adaptive_avg_pool2d(x, (32, 32)).flatten(2).transpose(1, 2)
        channels = torch.arange(768, device=x.device)
        ramp = torch.arange(1024, device=x.device).float()[None, :, None] / 1024
        patches = pooled[:, :, channels % 3] * (1 + channels.float()[None, None] / 768)
        patches = patches + torch.sin(channels.float()[None, None] * .07 + ramp * 5)
        cls = patches.mean(1, keepdim=True)
        tokens = torch.cat((cls, patches), 1) * self.weight
        if self.bad_shape:
            tokens = tokens[:, :-1]
        for block in self.blocks:
            tokens = block(tokens)
        if self.fail:
            raise RuntimeError('synthetic teacher failure')
        return tokens


class ToyStudent(nn.Module):
    """Trainable upstream -> actual C2f P4; component/batch-sum loss contract."""
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(3, 256, 1)] + [nn.Identity() for _ in range(17)]
                                   + [C2f(256, 256, n=1)])

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (40, 40))
        for module in self.model:
            x = module(x)
        return x

    def loss(self, batch, preds=None):
        x = self(batch['img']) if preds is None else preds
        scalar = x.square().mean() + 1
        items = torch.stack([scalar * (i + 1) for i in range(5)])
        return items * batch['img'].shape[0], items.detach()


def make_trainer(module, tmp_path, monkeypatch, batch_size=2):
    trainer = object.__new__(module.SpatialNoiseTrainer)
    trainer.model = ToyStudent().train()
    trainer.device = torch.device('cpu')
    trainer.args = SimpleNamespace(lr0=.001, multi_scale=0, resume=False, compile=False)
    trainer.save_dir = tmp_path
    trainer.epoch = 0
    trainer._batch_id = 0
    trainer._smoke_ratios = []
    trainer._shuffle_generator = torch.Generator().manual_seed(module.SEED)
    if module.KD_LOSS_VARIANT in ('relation', 'infonce'):
        trainer._relation_generator = torch.Generator().manual_seed(module.SEED)
    if module.LAM_AUTO_R is not None:
        trainer._lam_sp = module.LAM_SP
        trainer._lam_auto_ratios = []
        trainer._lam_auto_done = False
    trainer._teacher = FakeViT().eval().requires_grad_(False)
    trainer.optimizer = trainer.build_optimizer(trainer.model, name='SGD', lr=.001)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda e: .9 ** e)
    module._KD.update(calls=0, applied=0, sum=0.0)
    module._install_spatial_loss(trainer)
    raw = torch.randint(0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8)
    return trainer, raw


def test_original_sources_untouched():
    expected = 'eada6d2fe3cf580af6891767dcab25f296ae52ae3a9c6a6219cfcf6417d55118'
    for name in ('seed_sweep.py', 'seed_sweep_v1_reference.py'):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected


def test_legacy_functions_ast_unchanged():
    old = ast.parse((ROOT / 'seed_sweep.py').read_text())
    new = ast.parse((ROOT / 'seed_sweep_v2.py').read_text())
    names = ('SharedNoiseTrainer', 'teacher_cls', 'rkd_loss', 'load_teacher', '_sppf_hook')
    for name in names:
        a = next(n for n in old.body if getattr(n, 'name', None) == name)
        b = next(n for n in new.body if getattr(n, 'name', None) == name)
        assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


@pytest.mark.parametrize('exp', ['BL0', 'C1_CLEAN_RKD', 'A_NOISY', 'N1_MIXED'])
def test_legacy_one_batch_exact_regression(exp, monkeypatch):
    monkeypatch.setattr(nn.Module, 'cuda', lambda self, *a, **kw: self)
    def setup_stub(trainer, *args, **kwargs):
        trainer.optimizer = trainer.build_optimizer(trainer.model, name='SGD', lr=.001)
        trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda e: 1.)
    monkeypatch.setattr(SegmentationTrainer, '_setup_train', setup_stub)
    results = []
    for file in ('seed_sweep_v1_reference.py', 'seed_sweep_v2.py'):
        module = load_module(exp, file)
        monkeypatch.setattr(module, 'load_teacher', lambda: FakeViT().eval().requires_grad_(False))
        module.set_all_seeds(77)
        trainer = object.__new__(module.SharedNoiseTrainer)
        trainer.device = torch.device('cpu')
        trainer.args = SimpleNamespace(lr0=.001, multi_scale=0)
        trainer.model = SegmentationModel('yolov8s-seg.yaml', nc=2, verbose=False).train()
        trainer.model.args = get_cfg(overrides={'overlap_mask': False})
        trainer._setup_train()
        raw = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
        batch = trainer.preprocess_batch({'img': raw.clone(), 'batch_idx': torch.empty(0),
            'cls': torch.empty(0, 1), 'bboxes': torch.empty(0, 4), 'masks': torch.empty(0, 64, 64)})
        noisy_indices = torch.nonzero((batch['img'] != raw.float() / 255).flatten(1).any(1)).flatten()
        # Class method bypasses only the instance KD wrapper to measure L_det.
        det, _ = type(trainer.model).loss(trainer.model, batch)
        total, _ = trainer.model.loss(batch)
        result = (batch['img'].detach().clone(), noisy_indices, det.detach().clone(),
                  total.detach().clone(), module._KD.copy(),
                  [(g['lr'], g['weight_decay'], len(g['params'])) for g in trainer.optimizer.param_groups])
        results.append(result)
    for a, b in zip(results[0][:4], results[1][:4]):
        assert torch.equal(a, b)
    assert results[0][4:] == results[1][4:]


@pytest.mark.parametrize('exp,norm,shuffle,lam', [('S0','imagenet',False,0), ('S1','imagenet',False,.5),
                                                ('S2','imagenet',True,.5), ('S1r','none',False,.5)])
def test_condition_table(exp, norm, shuffle, lam):
    m = load_module(exp)
    assert (m.KD_MODE, m.USE_KD, m.USE_NOISE) == ('spatial', True, True)
    assert (m.TEACHER_NORM, m.SHUFFLE, m.LAM_SP) == (norm, shuffle, lam)
    assert (m.KD_LAYER, m.MASK_TOP) == (18, .01)
    if exp == 'S0':
        assert load_module(exp, LAM_SP=100).LAM_SP == 0


def test_teacher_none_input_equals_v1(kd):
    clean = torch.rand(2,3,51,63)
    a, b = FakeViT(), FakeViT()
    reference = load_module('N1_MIXED', 'seed_sweep_v1_reference.py')
    reference.teacher_cls(a, clean)
    kd.raw_teacher_patches(b, clean, 'none')
    assert torch.equal(a.seen[0], b.seen[0])
    expected = F.interpolate(clean, (448,448), mode='bilinear', align_corners=False)
    assert torch.equal(kd.prepare_teacher_input(clean, 'none'), expected)
    mean = expected.new_tensor([.485,.456,.406])[None,:,None,None]
    std = expected.new_tensor([.229,.224,.225])[None,:,None,None]
    assert torch.equal(kd.prepare_teacher_input(clean,'imagenet'), (expected-mean)/std)


def test_teacher_microbatch_shape_and_freeze(kd):
    clean = torch.rand(5,3,48,48)
    teacher = FakeViT().train()
    raw = kd.raw_teacher_patches(teacher, clean, micro_batch=2)
    assert raw.shape == (5,768,32,32)
    assert teacher.batch_sizes == [2,2,1]
    assert raw.dtype == torch.float32 and not raw.requires_grad
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    full = kd.raw_teacher_patches(teacher, clean, micro_batch=5)
    torch.testing.assert_close(raw, full, rtol=0, atol=0)
    assert not teacher.blocks[11]._forward_hooks


@pytest.mark.parametrize('bad_shape,fail', [(True,False), (False,True)])
def test_teacher_hook_removed_on_error(kd, bad_shape, fail):
    teacher = FakeViT(bad_shape, fail)
    with pytest.raises(RuntimeError):
        kd.raw_teacher_patches(teacher, torch.rand(1,3,40,40))
    assert not teacher.blocks[11]._forward_hooks


def test_teacher_rejects_wrong_block_count(kd):
    teacher = FakeViT()
    teacher.blocks = teacher.blocks[:-1]
    with pytest.raises(RuntimeError, match='12 blocks'):
        kd.raw_teacher_patches(teacher, torch.rand(1,3,40,40))


def test_cached_real_dinov2_or_fake(kd):
    hub = Path(torch.hub.get_dir())
    cached = list(hub.glob('facebookresearch_dinov2_*')) and (hub/'checkpoints/dinov2_vitb14_pretrain.pth').is_file()
    teacher = kd.load_spatial_teacher('cpu') if cached else FakeViT()
    raw, mask = kd.teacher_patches(teacher, torch.rand(1,3,64,64))
    assert raw.shape == (1,768,32,32) and mask.sum() == 1014
    assert not raw.requires_grad and all(p.grad is None for p in teacher.parameters())
    assert not teacher.blocks[11]._forward_hooks


def test_grid_coordinate_ramp_asymmetric_mark(kd):
    kd.check_grid_order()
    tokens = torch.zeros(2,1025,768)
    tokens[:,0] = -999
    tokens[:,1:,0] = torch.arange(1024)
    grid = kd.tokens_to_grid(tokens)
    assert grid[1,0,7,23] == 7*32+23
    assert not (grid == -999).any()


def test_raw_mask_exact_top10_before_normalization(kd):
    raw = torch.arange(1,1025).float()[None,None,None,:].expand(2,768,1,1024).reshape(2,768,32,32)
    mask = kd.anomaly_mask(raw)
    assert torch.equal(mask.flatten(1).sum(1), torch.tensor([1014,1014]))
    assert not mask.flatten(1)[:,-10:].any()
    assert mask.flatten(1)[:,:-10].all()
    # All channel-constant vectors become zero after LN: this would mask wrong positions.
    assert not torch.equal(mask, kd.anomaly_mask(kd.normalize_spatial(raw)))


def test_common_mask_across_s1_s2_s1r(kd):
    teacher = FakeViT()
    clean = torch.rand(2,3,40,40)
    raw1, mask1 = kd.teacher_patches(teacher, clean, 'imagenet')
    rawr, maskr = kd.teacher_patches(teacher, clean, 'none')
    assert torch.equal(mask1, maskr)
    assert not torch.equal(raw1, rawr)


@pytest.mark.parametrize('mode', ['bilinear_antialias', 'area'])
def test_adapter_shapes_params_fp32_and_backward(kd, mode):
    adapter = kd.SpatialAdapter(mode)
    assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == 787712
    assert not any(isinstance(x, nn.BatchNorm2d) for x in adapter.modules())
    x = torch.randn(1,256,40,40, requires_grad=True)
    shapes = {}
    handles = [mod.register_forward_hook(lambda m,i,o,n=name: shapes.update({n:tuple(o.shape)}))
               for name, mod in [('conv3',adapter.conv3), ('gn',adapter.gn), ('conv1',adapter.conv1)]]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = adapter(x)
    assert shapes == {'conv3':(1,256,40,40), 'gn':(1,256,40,40), 'conv1':(1,768,40,40)}
    assert output.shape == (1,768,32,32) and output.dtype == torch.float32
    torch.testing.assert_close(output.norm(dim=1), torch.ones(1,32,32), atol=1e-6, rtol=1e-6)
    kd.spatial_kd_loss(output, torch.randn_like(output), torch.ones(1,32,32,dtype=torch.bool)).backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0
    assert all(p.grad is not None and p.grad.norm() > 0 for p in adapter.parameters())
    for handle in handles:
        handle.remove()
    with pytest.raises(RuntimeError, match='P4'):
        adapter(torch.rand(1,256,20,20))


def test_actual_yolo_p4_shape(kd):
    model = SegmentationModel('yolov8s-seg.yaml', nc=2, verbose=False).eval()
    assert type(model.model[18]).__name__ == 'C2f'
    out = []
    handle = model.model[18].register_forward_hook(lambda m,i,o: out.append(o.shape))
    with torch.no_grad():
        model(torch.zeros(1,3,640,640))
    handle.remove()
    assert out == [torch.Size((1,256,40,40))]


def direction_fixture():
    # Every location has a distinct, LN-nondegenerate direction; roll changes correspondence.
    return torch.tensor([[[[1.,0.],[0.,-1.]], [[0.,1.],[-1.,0.]], [[-1.,-1.],[1.,1.]]]])


def test_spatial_loss_identical_opposite_shuffle_range(kd):
    x = direction_fixture()
    mask = torch.ones(1,2,2,dtype=torch.bool)
    assert kd.spatial_kd_loss(x,x,mask).item() == pytest.approx(0,abs=2e-7)
    assert kd.spatial_kd_loss(x,-x,mask).item() == pytest.approx(2,abs=2e-7)
    shuffled = x.flatten(2).roll(1,2).reshape_as(x)
    assert kd.spatial_kd_loss(x,shuffled,mask) > 0
    for _ in range(4):
        loss = kd.spatial_kd_loss(torch.randn(3,8,4,4), torch.randn(3,8,4,4), torch.ones(3,4,4,dtype=torch.bool))
        assert torch.isfinite(loss) and 0 <= loss <= 2


def test_zero_and_spatial_constant_loss(kd):
    zero = torch.zeros(1,3,2,2)
    mask = torch.ones(1,2,2,dtype=torch.bool)
    assert kd.spatial_kd_loss(zero, zero, mask) == 1
    # Spatial constancy is shuffle-invariant; channel constancy gives LN zero -> loss 1.
    channel_constant = torch.full_like(zero, 3)
    assert kd.spatial_kd_loss(channel_constant, channel_constant, mask) == 1
    spatial_constant = torch.tensor([1.,2.,-1.])[None,:,None,None].expand_as(zero)
    perm = spatial_constant.flatten(2).roll(1,2).reshape_as(zero)
    assert kd.spatial_kd_loss(spatial_constant,perm,mask) == pytest.approx(0,abs=2e-7)


def test_relation_loss_identity_collapse_and_permutation(kd):
    target = torch.tensor([[[[1., 0., -1., 0.]], [[0., 1., 0., -1.]]]])
    mask = torch.ones(1, 1, 4, dtype=torch.bool)
    generator = torch.Generator().manual_seed(7)
    assert kd.relation_spatial_loss(target, target, mask, generator, 4) == pytest.approx(0, abs=1e-7)
    constant = torch.ones_like(target)
    assert kd.relation_spatial_loss(constant, target, mask, torch.Generator().manual_seed(7), 4) > 0
    # 위치 섞기가 관계 행렬을 바꾸는 비대칭 배치 (원 배치는 90도 회전 대칭이라 섞어도 행렬이 같았다)
    asym = torch.tensor([[[[1., 0., 0., 2.]], [[0., 1., 0., 0.]], [[0., 0., 1., -1.]]]])
    shuffled = asym[..., [1, 0, 2, 3]]
    assert kd.relation_spatial_loss(asym, shuffled, mask, torch.Generator().manual_seed(7), 4) > 0


def test_infonce_identity_and_spatial_collapse(kd):
    target = torch.tensor([[[[1., 0., -1., 0.]], [[0., 1., 0., -1.]]]])
    mask = torch.ones(1, 1, 4, dtype=torch.bool)
    same = kd.infonce_spatial_loss(target, target, mask, torch.Generator().manual_seed(3), 4, .1)
    collapsed = kd.infonce_spatial_loss(torch.ones_like(target), target, mask,
                                        torch.Generator().manual_seed(3), 4, .1)
    assert same < .01
    assert collapsed == pytest.approx(float(torch.log(torch.tensor(4.))), abs=1e-6)


def test_fgcenter_weights_and_no_box_equivalence(kd):
    target = torch.tensor([[[[1., 0., -1., 0.]], [[0., 1., 0., -1.]]]])
    student = target.roll(1, 3)
    mask = torch.ones(1, 1, 4, dtype=torch.bool)
    weights = kd.foreground_weights({}, 1, target.device, 1, 4, 3.)
    assert kd.fgcenter_spatial_loss(student, target, mask, weights) == pytest.approx(
        kd.centered_spatial_loss(student, target, mask), abs=1e-7)
    small = {'bboxes': torch.tensor([[.51, .5, .001, .001]]), 'batch_idx': torch.tensor([0])}
    weights = kd.foreground_weights(small, 1, target.device, 1, 4, 3.)
    assert torch.equal(weights, torch.tensor([[[1., 1., 3., 1.]]]))


def test_masked_positions_ignored_and_group_weighting(kd):
    x = direction_fixture().repeat(3,1,1,1)
    y = x.clone()
    mask = torch.ones(3,2,2,dtype=torch.bool)
    mask[:,0,0] = False
    y[:,:,0,0] = -x[:,:,0,0]
    assert kd.spatial_kd_loss(x,y,mask) == pytest.approx(0,abs=2e-7)
    y[2] = -x[2]
    loss, stats = kd.spatial_kd_loss(x,y,mask,torch.tensor([False,False,True]),True)
    assert loss == pytest.approx(2/3, abs=2e-7)
    assert stats['clean'] == pytest.approx(0,abs=2e-7)
    assert stats['noisy'] == pytest.approx(2,abs=2e-7)
    assert loss == pytest.approx((2*stats['clean']+stats['noisy'])/3,abs=2e-7)
    with pytest.raises(ValueError):
        kd.spatial_kd_loss(x,y,torch.zeros_like(mask))


def test_shuffle_valid_set_channels_images_and_rng(kd):
    target = torch.arange(2*5*1024).reshape(2,5,32,32)
    mask = torch.ones(2,32,32,dtype=torch.bool)
    mask.flatten(1)[:,:10] = False
    original_mask = mask.clone()
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(24)
    state = torch.get_rng_state().clone()
    a = kd.shuffle_teacher_tokens(target,mask,generator)
    first_state = generator.get_state().clone()
    b = kd.shuffle_teacher_tokens(target,mask,generator)
    assert torch.equal(torch.get_rng_state(), state)
    assert not torch.equal(first_state,generator.get_state())
    assert not torch.equal(a,b) and not torch.equal(a,target)
    assert torch.equal(mask,original_mask)
    for image in range(2):
        flat = mask[image].flatten()
        assert torch.equal(a[image].flatten(1)[:,~flat],target[image].flatten(1)[:,~flat])
        # Recover the permutation using channel 0, then verify all channels use it.
        original = target[image].flatten(1)
        result = a[image].flatten(1)
        permutation = (result[0] - original[0,0]).long()
        assert torch.equal(result,original[:,permutation])
        assert torch.equal(result[0].sort().values,original[0].sort().values)
    replay = kd.shuffle_teacher_tokens(target,mask,torch.Generator().manual_seed(0))
    assert torch.equal(a,replay)


@pytest.mark.parametrize('reduction',['batch_sum','image_mean'])
def test_loss_added_once_and_batch_invariant(kd,reduction):
    for B in (1,2,8):
        scale = B if reduction=='batch_sum' else 1
        det = torch.tensor([1.,2.,3.,4.,5.],requires_grad=True)*scale
        spatial = torch.tensor(.4,requires_grad=True)
        total,d_img = kd.add_spatial_loss(det,spatial,B,.5,reduction)
        assert total.ndim == 0 and d_img == 15
        assert total.item() == pytest.approx((15+.5*.4)*scale)
        g, = torch.autograd.grad(total,spatial)
        assert g.item()/scale == .5
        zero,_ = kd.add_spatial_loss(det,spatial,B,0,reduction)
        assert torch.equal(zero,det.sum())


@pytest.mark.parametrize('exp',['S0','S1','S2','S1r'])
def test_new_trainer_one_batch_gradients_logs_and_cache(exp,tmp_path,monkeypatch):
    m = load_module(exp)
    trainer,raw = make_trainer(m,tmp_path,monkeypatch)
    torch.manual_seed(103)
    state = torch.get_rng_state().clone()
    expected_idx = torch.randperm(2)[:1]
    expected_noise = torch.randn(1,3,64,64)*m.NOISE_SIGMA
    expected = raw.float()/255
    clean = expected.clone()
    expected[expected_idx] = (expected[expected_idx]+expected_noise).clamp(0,1)
    expected_state = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    batch = trainer.preprocess_batch({'img':raw})
    assert torch.equal(batch['img'],expected)
    assert torch.equal(torch.get_rng_state(),expected_state)
    assert m._SPATIAL_CACHE['noisy'].sum()==1
    assert torch.equal(trainer._teacher.seen[0],m.prepare_teacher_input(clean,m.TEACHER_NORM))
    adapter_ids = {id(p) for p in trainer._adapter.parameters()}
    optimizer_ids = [id(p) for g in trainer.optimizer.param_groups for p in g['params']]
    assert all(optimizer_ids.count(p)==1 for p in adapter_ids)
    assert not any(id(p) in optimizer_ids for p in trainer._teacher.parameters())
    loss, items = trainer.model.loss(batch)
    stats = json.loads((tmp_path/'spatial_kd.jsonl').read_text().splitlines()[-1])
    expected_total = stats['L_det'] + m.LAM_SP*stats['L_sp']*2
    assert loss.item() == pytest.approx(expected_total,rel=1e-6)
    assert stats['adapter_kd_grad_norm']>0 and stats['upstream_kd_grad_norm']>0
    assert m._KD['calls']==m._KD['applied']==1
    assert not m._SPATIAL_CACHE
    assert all(p.grad is None for p in trainer._adapter.parameters())  # diagnostic has no side effects
    loss.backward()
    assert all(p.grad is None for p in trainer._teacher.parameters())
    if exp=='S0':
        assert torch.equal(loss.detach(),torch.tensor(stats['L_det']))
        assert all(p.grad is None for p in trainer._adapter.parameters())
    else:
        assert sum(p.grad.norm() for p in trainer._adapter.parameters())>0
    trainer.optimizer.step()
    trainer.scheduler.step()  # crosses an epoch boundary with the adapter group present
    assert len(trainer.optimizer.param_groups)==len(trainer.scheduler.base_lrs)
    m.require_spatial_kd_applied()
    with pytest.raises(RuntimeError,match='optimizer rebuild'):
        trainer.build_optimizer(trainer.model,name='SGD')
    with pytest.raises(RuntimeError,match='stale'):
        trainer.model.loss(batch)


def test_default_adapter_group_and_lambda_are_unchanged(kd,tmp_path,monkeypatch):
    trainer,_ = make_trainer(kd,tmp_path,monkeypatch)
    adapter = trainer.optimizer.param_groups[-1]
    assert adapter['lr'] == pytest.approx(trainer.args.lr0)
    assert adapter['weight_decay'] == pytest.approx(0.)
    assert kd.lam_schedule_factor(0, 10) == 1
    assert kd.lam_schedule_factor(10, 10) == 1


def test_adapter_lr_multiplier_and_weight_decay_follow_scheduler(tmp_path,monkeypatch):
    m = load_module('S1', ADAPTER_LR_MULT=1, ADAPTER_WD=5e-4)
    trainer,_ = make_trainer(m,tmp_path,monkeypatch)
    body_lr = trainer.optimizer.param_groups[0]['lr']
    adapter = trainer.optimizer.param_groups[-1]
    assert adapter['lr'] == pytest.approx(body_lr)
    assert adapter['weight_decay'] == pytest.approx(5e-4)
    trainer.optimizer.step()
    trainer.scheduler.step()
    assert adapter['lr'] == pytest.approx(trainer.optimizer.param_groups[0]['lr'])


def test_prealign_detaches_kd_upstream_then_restores_it(tmp_path,monkeypatch):
    m = load_module('S1', ADAPTER_PREALIGN_STEPS=1, DIAG_EVERY=1)
    trainer,raw = make_trainer(m,tmp_path,monkeypatch)
    first = trainer.preprocess_batch({'img': raw})
    trainer.model.loss(first)
    before = json.loads((tmp_path/'spatial_kd.jsonl').read_text().splitlines()[-1])
    assert before['prealign'] is True
    assert before['adapter_kd_grad_norm'] > 0
    assert before['upstream_kd_grad_norm'] == 0
    second = trainer.preprocess_batch({'img': raw})
    trainer.model.loss(second)
    after = json.loads((tmp_path/'spatial_kd.jsonl').read_text().splitlines()[-1])
    assert after['prealign'] is False
    assert after['upstream_kd_grad_norm'] > 0


def test_last_batch_missing_clean_log(tmp_path,monkeypatch):
    m=load_module('S1')
    trainer,raw=make_trainer(m,tmp_path,monkeypatch,batch_size=1)
    batch=trainer.preprocess_batch({'img':raw})
    loss,_=trainer.model.loss(batch)
    assert torch.isfinite(loss)
    stats=json.loads((tmp_path/'spatial_kd.jsonl').read_text())
    assert stats['clean'] is None and stats['noisy'] is not None and stats['noisy_count']==1


def test_hook_eval_does_not_overwrite_and_missing_feature_fails(kd,tmp_path,monkeypatch):
    trainer,raw=make_trainer(kd,tmp_path,monkeypatch)
    batch=trainer.preprocess_batch({'img':raw})
    pred=trainer.model(batch['img'])
    cached=kd._SPATIAL_CACHE['student']
    trainer.model.eval()
    with torch.no_grad():
        trainer.model(batch['img'])
    assert kd._SPATIAL_CACHE['student'] is cached
    trainer.model.train()
    kd._SPATIAL_CACHE.pop('student')
    with pytest.raises(RuntimeError,match='P4 feature missing'):
        trainer.model.loss(batch,pred)
    # A new preprocess invalidates the older batch, even at the same shape.
    trainer.preprocess_batch({'img':raw})
    with pytest.raises(RuntimeError,match='stale'):
        trainer.model.loss(batch,pred)


def test_reject_wrong_student_and_resume(kd,tmp_path,monkeypatch):
    trainer,raw=make_trainer(kd,tmp_path,monkeypatch)
    trainer.args.resume=True
    with pytest.raises(RuntimeError,match='resume'):
        trainer._setup_train()
    bad=object.__new__(kd.SpatialNoiseTrainer)
    bad.model=ToyStudent()
    bad.model.model[18]=nn.Identity()
    with pytest.raises(RuntimeError,match='C2f'):
        bad.build_optimizer(bad.model,name='SGD')


def test_missing_kd_and_required_train_args_gate(kd):
    kd._KD.update(calls=1,applied=0)
    with pytest.raises(AssertionError,match='count mismatch'):
        kd.require_spatial_kd_applied()
    measured=valid_measurements(kd)
    measured.pop('lr0')
    measured.pop('amp')
    errors=kd._declared_vs_measured(measured)
    assert any('lr0' in x for x in errors) and any('amp' in x for x in errors)
    measured.update(lr0=.001,amp=True)
    assert kd._declared_vs_measured(measured)==[]
    measured['batch']=4
    assert any('batch' in x for x in kd._declared_vs_measured(measured))


def test_manifest_normalization_and_entry_gate(kd,tmp_path,monkeypatch):
    del kd.__dict__['__SRC_PATH__']
    with pytest.raises(RuntimeError,match='__SRC_PATH__'):
        kd._entry_source()
    kd.__SRC_PATH__=str(ROOT/'seed_sweep.py')
    with pytest.raises(RuntimeError,match='hash differs'):
        kd._entry_source()
    kd.__SRC_PATH__=str(ROOT/'seed_sweep_v2.py')
    assert kd._entry_source()[1] == kd._sha256(kd.__SRC_PATH__)
    condition=dict(kd_mode='spatial',shuffle=False,teacher_norm='imagenet',teacher_input_normalized=False,
        kd_layer=18,lam_sp=.5,mask_top=.01,interp_mode='bilinear_antialias',adapter_params=787712,
        kd_loss_variant='plain',relation_k=256,infonce_k=256,infonce_tau=.1,fg_alpha=3.,lam_auto_r=None,
        mask_reference_norm='imagenet',det_reduction='batch_sum',label_prefix_fix=True,skip_val_loss=True,
        adapter_lr_mult=None,adapter_wd=None,adapter_prealign_steps=0,lam_schedule='const',
        lam_warmup_frac=.1,lam_end_frac=.1)
    assert any('teacher_input_normalized' in x for x in kd._spatial_manifest_errors(condition))
    condition['teacher_input_normalized']=True
    assert kd._spatial_manifest_errors(condition)==[]


def test_lam_auto_recalibrates_once_on_step_51():
    m = load_module('S1', LAM_AUTO_R=.2, LAM_AUTO_STEPS=50)
    trainer = SimpleNamespace(_lam_sp=m.LAM_SP, _lam_auto_ratios=[], _lam_auto_done=False)
    for step in range(1, 51):
        assert m._lam_auto_step(trainer, .4, step) is None
    event = m._lam_auto_step(trainer, 99., 51)
    assert trainer._lam_sp == pytest.approx(.5 * .2 / .4)
    assert event == pytest.approx({'target': .2, 'measured_R': .4, 'lam_before': .5,
                                   'lam_after': .25, 'step': 51})
    assert m._lam_auto_step(trainer, .1, 52) is None
    assert trainer._lam_sp == pytest.approx(.25)


def test_lambda_schedules_and_auto_calibration_use_base_lambda():
    warm = load_module('S1', LAM_SCHEDULE='warmup', LAM_WARMUP_FRAC=.2)
    assert warm.lam_schedule_factor(0, 10) == 0
    assert warm.lam_schedule_factor(1, 10) == pytest.approx(.5)
    assert warm.lam_schedule_factor(10, 10) == 1
    cosine = load_module('S1', LAM_SCHEDULE='warmup_cos', LAM_WARMUP_FRAC=.2, LAM_END_FRAC=.1)
    assert cosine.lam_schedule_factor(0, 10) == 0
    assert cosine.lam_schedule_factor(1, 10) == pytest.approx(.5)
    assert cosine.lam_schedule_factor(2, 10) == 1
    assert cosine.lam_schedule_factor(10, 10) == pytest.approx(.1)
    m = load_module('S1', LAM_AUTO_R=.2, LAM_AUTO_STEPS=2,
                    LAM_SCHEDULE='warmup', LAM_WARMUP_FRAC=.5)
    trainer = SimpleNamespace(_lam_sp=m.LAM_SP, _lam_auto_ratios=[], _lam_auto_done=False)
    # Effective R would be .1 and .2 here; the wrapper converts both back to
    # the base-lambda R=.4 before collecting them.
    assert m._lam_auto_step(trainer, .4, 1) is None
    assert m._lam_auto_step(trainer, .4, 2) is None
    m._lam_auto_step(trainer, .4, 3)
    assert trainer._lam_sp == pytest.approx(.25)


@pytest.mark.parametrize('settings', [
    {'ADAPTER_LR_MULT': -1}, {'ADAPTER_WD': -1}, {'ADAPTER_PREALIGN_STEPS': -1},
    {'LAM_SCHEDULE': 'unknown'}, {'LAM_WARMUP_FRAC': -0.1}, {'LAM_END_FRAC': -0.1},
])
def test_strategy_options_reject_invalid_values(settings):
    with pytest.raises(AssertionError):
        load_module('S1', **settings)


@pytest.mark.parametrize('variant', ['relation', 'infonce', 'fgcenter'])
def test_new_variants_tiny_batch_backward(variant, tmp_path, monkeypatch):
    m = load_module('S1', KD_LOSS_VARIANT=variant, RELATION_K=8, INFONCE_K=8)
    trainer, raw = make_trainer(m, tmp_path, monkeypatch)
    batch = trainer.preprocess_batch({'img': raw})
    loss, _ = trainer.model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert sum(p.grad.norm() for p in trainer._adapter.parameters()) > 0
    assert sum(p.grad.norm() for p in trainer.model.model[0].parameters()) > 0


@pytest.mark.parametrize('ratios,expected,required',[([.5],.5,False),([.6,.4],.5,False),([.6,.8],.5*.2/.7,True)])
def test_smoke_lambda_rule(kd,ratios,expected,required):
    result=kd.smoke_lambda_recommendation(ratios,.5)
    assert result['lam_sp_recommended']==pytest.approx(expected)
    assert result['recalibration_required']==required


def test_save_student_and_ema_reload_without_script(kd,tmp_path,monkeypatch):
    model=SegmentationModel('yolov8s-seg.yaml',nc=2,verbose=False).train()
    model.model[18].register_forward_hook(kd._p4_hook)
    # Both targets deliberately carry unpickleable closures.
    model.loss=lambda *args: None
    ema=copy.deepcopy(model).eval()
    trainer=object.__new__(kd.SpatialNoiseTrainer)
    trainer.model=model
    trainer.ema=SimpleNamespace(ema=ema,updates=0)
    trainer.optimizer=torch.optim.SGD(model.parameters(),lr=.001)
    trainer.scaler=torch.amp.GradScaler('cuda',enabled=False)
    trainer.epoch=0
    trainer.best_fitness=trainer.fitness=0.
    trainer.metrics={}
    trainer.args=get_cfg()
    trainer.wdir=tmp_path/'weights'
    trainer.last=trainer.wdir/'last.pt'
    trainer.best=trainer.wdir/'best.pt'
    trainer.save_period=-1
    trainer.read_results_csv=lambda: {}
    assert trainer.save_model()
    assert 'loss' in model.__dict__ and 'loss' in ema.__dict__
    assert model.model[18]._forward_hooks and ema.model[18]._forward_hooks
    assert 'student-only checkpoint OK' in kd.verify_student_checkpoint(trainer.best)
    checkpoint=torch.load(trainer.best,map_location='cpu',weights_only=False)
    assert not checkpoint['ema'].model[18]._forward_hooks
    assert 'loss' not in checkpoint['ema'].__dict__


def test_save_restores_state_on_exception(kd,tmp_path,monkeypatch):
    trainer=object.__new__(kd.SpatialNoiseTrainer)
    trainer.model=ToyStudent()
    trainer.model.loss=lambda *a: None
    trainer.model.model[18].register_forward_hook(kd._p4_hook)
    trainer.ema=SimpleNamespace(ema=copy.deepcopy(trainer.model))
    def fail(self):
        assert 'loss' not in self.model.__dict__ and 'loss' not in self.ema.ema.__dict__
        raise OSError('synthetic save failure')
    monkeypatch.setattr(SegmentationTrainer,'save_model',fail)
    with pytest.raises(OSError,match='synthetic save failure'):
        trainer.save_model()
    assert 'loss' in trainer.model.__dict__ and trainer.model.model[18]._forward_hooks
    assert 'loss' in trainer.ema.ema.__dict__ and trainer.ema.ema.model[18]._forward_hooks


def test_reexec_s0_then_s1_retains_shared_lambda():
    path=ROOT/'seed_sweep_v2.py'
    namespace={'EXP':'S0','__SRC_PATH__':str(path),'LAM_SP_SHARED':.31}
    code=compile(path.read_text(),str(path),'exec')
    exec(code,namespace)
    assert namespace['LAM_SP']==0
    namespace['EXP']='S1'
    exec(code,namespace)
    assert namespace['LAM_SP']==.31


def test_full_spatial_setup_registers_before_scheduler(kd,tmp_path,monkeypatch):
    trainer=object.__new__(kd.SpatialNoiseTrainer)
    trainer.model=ToyStudent().train()
    trainer.device=torch.device('cpu')
    trainer.args=get_cfg()
    trainer.save_dir=tmp_path
    trainer.epoch=0
    trainer.callbacks={'on_train_epoch_end':[]}
    def base_setup(self):
        self.optimizer=self.build_optimizer(self.model,name='SGD',lr=.001)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda e:1.)
        self.start_epoch=0
        self.ema=SimpleNamespace(ema=copy.deepcopy(self.model).eval())
    monkeypatch.setattr(SegmentationTrainer,'_setup_train',base_setup)
    monkeypatch.setattr(kd,'load_spatial_teacher',lambda device:FakeViT().eval().requires_grad_(False))
    trainer._setup_train()
    assert len(trainer.optimizer.param_groups)==len(trainer.scheduler.base_lrs)==4
    assert kd._SPATIAL_RUN['optimizer_actual']=='SGD'
    assert kd._SPATIAL_RUN['scheduler_group_count']==4
    assert trainer._teacher is not None and 'loss' in trainer.model.__dict__
    assert 'loss' not in trainer.ema.ema.__dict__
    assert trainer.callbacks['on_train_epoch_end']==[kd._spatial_epoch_end]
    assert kd._SPATIAL_RUN['entry_sha256']==kd._sha256(ROOT/'seed_sweep_v2.py')


def test_normal_batch8_mask_and_microbatch(kd,tmp_path,monkeypatch):
    trainer,raw=make_trainer(kd,tmp_path,monkeypatch,batch_size=8)
    trainer.preprocess_batch({'img':raw})
    assert kd._SPATIAL_CACHE['noisy'].sum()==4
    assert trainer._teacher.batch_sizes==[2,2,2,2]
    assert torch.equal(kd._SPATIAL_CACHE['mask'].flatten(1).sum(1),torch.full((8,),1014))


def test_train_mode_ema_cannot_overwrite_student_cache(kd,tmp_path,monkeypatch):
    trainer,raw=make_trainer(kd,tmp_path,monkeypatch)
    ema=copy.deepcopy(trainer.model).train()
    batch=trainer.preprocess_batch({'img':raw})
    trainer.model(batch['img'])
    before=kd._SPATIAL_CACHE['student']
    with torch.no_grad():
        ema(batch['img'])
    assert kd._SPATIAL_CACHE['student'] is before


def test_gate_detects_training_loss_bypassing_wrapper(kd):
    kd._KD.update(calls=3,applied=3)
    kd._SPATIAL_RUN['training_batches']=4
    with pytest.raises(AssertionError,match='count mismatch'):
        kd.require_spatial_kd_applied()


def valid_measurements(m):
    return dict(epochs=m.EPOCHS,batch=m.BATCH,imgsz=m.IMGSZ,seed=m.SEED,lr0=.001,lrf=.1,
                optimizer='auto',workers=m.WORKERS,fraction=m.FRACTION,epochs_actually_recorded=m.EPOCHS,
                warmup_epochs=.1,warmup_bias_lr=0.,amp=True,deterministic=True,
                weight_decay=.0005,mosaic=1.,resume=False,compile=False)


def manifest_fixture(m,tmp_path,monkeypatch):
    measured=valid_measurements(m)
    monkeypatch.setattr(m,'_ckpt_train_args',lambda p:dict(measured))
    weights=tmp_path/'weights'
    weights.mkdir()
    (weights/'best.pt').write_bytes(b'checkpoint fixture; reader is mocked')
    (weights/'last.pt').write_bytes(b'checkpoint fixture; reader is mocked')  # EVAL_CKPT 기본 last (정정 4)
    (tmp_path/'results.csv').write_text('epoch\n1\n')
    (tmp_path/'spatial_kd.jsonl').write_text('{}\n')
    m._SPATIAL_RUN.update(adapter_params=787712,optimizer_actual='SGD',optimizer_group_lrs=[.001]*4,
                          checkpoint_reload='student-only checkpoint OK',training_batches=1,
                          entry_sha256=m._sha256(m.__SRC_PATH__))
    m._KD.update(calls=1,applied=1,sum=.2)
    return measured


@pytest.mark.parametrize('exp',['S0','S1','S2','S1r'])
def test_manifest_expansion_and_source_hashes(exp,tmp_path,monkeypatch):
    m=load_module(exp)
    manifest_fixture(m,tmp_path,monkeypatch)
    result=Path(m.write_manifest(tmp_path,{},str(ROOT/'SPEC.md')))
    doc=json.loads(result.read_text())
    assert doc['status']=='completed' and doc['schema_ver']=='2.1'
    assert m._spatial_manifest_errors(doc['condition'])==[]
    assert doc['condition']['teacher_input_normalized']==(exp!='S1r')
    assert doc['code']['entry_script_sha256']==m._sha256(doc['code']['copy_beside_run'])
    assert doc['env']['torch'] and doc['env']['ultralytics'] and 'gpu' in doc['env']
    assert doc['spatial_runtime']['optimizer_actual']=='SGD'
    assert doc['artifacts']['spatial_kd_log_sha256']


def test_manifest_writes_invalid_evidence_for_missing_key(kd,tmp_path,monkeypatch):
    measured=manifest_fixture(kd,tmp_path,monkeypatch)
    measured['lr0']=None
    with pytest.raises(AssertionError,match='lr0'):
        kd.write_manifest(tmp_path,{},str(ROOT/'SPEC.md'))
    doc=json.loads((tmp_path/'manifest.json').read_text())
    assert doc['status']=='invalid'
    assert any('lr0' in x for x in doc['declared_vs_measured_mismatch'])


@pytest.mark.parametrize('failure',['copy','hash','source_changed'])
def test_source_copy_failures_are_fatal(kd,tmp_path,monkeypatch,failure):
    manifest_fixture(kd,tmp_path,monkeypatch)
    if failure=='copy':
        def fail(*a,**kw):
            raise OSError('synthetic copy failure')
        monkeypatch.setattr(kd.shutil,'copy',fail)
        expected=OSError
    elif failure=='hash':
        original=kd._sha256
        monkeypatch.setattr(kd,'_sha256',lambda p:'wrong' if Path(p).name=='seed_sweep.used.py' else original(p))
        expected=RuntimeError
    else:
        kd._SPATIAL_RUN['entry_sha256']='old source'
        expected=RuntimeError
    with pytest.raises(expected):
        kd.write_manifest(tmp_path,{},str(ROOT/'SPEC.md'))


def test_smoke_report_only_epoch0_s1_seed0(tmp_path):
    for exp,seed,fraction,epoch,expected in [('S1',0,.05,0,True),('S1',0,.05,1,False),
        ('S1',1,.05,0,False),('S2',0,.05,0,False),('S1',0,1.,0,False)]:
        m=load_module(exp,SEED=seed,FRACTION=fraction)
        directory=tmp_path/f'{exp}_{seed}_{fraction}_{epoch}'
        directory.mkdir()
        trainer=SimpleNamespace(epoch=epoch,save_dir=directory,_smoke_ratios=[.6,.8])
        m._spatial_epoch_end(trainer)
        assert (directory/'lambda_smoke.json').exists()==expected
        assert m.LAM_SP==.5  # recommendation never changes the running experiment


def test_patch_embedding_grid_rejects_transposition(kd):
    class Embedding(nn.Module):
        def __init__(self,transpose):
            super().__init__()
            self.proj=nn.Conv2d(3,768,14,stride=14)
            self.norm=nn.Identity()
            self.transpose=transpose
        def forward(self,x):
            x=self.proj(x)
            if self.transpose:
                x=x.transpose(2,3)
            return x.flatten(2).transpose(1,2)
    good=SimpleNamespace(patch_embed=Embedding(False))
    kd.check_teacher_patch_order(good,'cpu')
    bad=SimpleNamespace(patch_embed=Embedding(True))
    with pytest.raises(RuntimeError,match='row-major'):
        kd.check_teacher_patch_order(bad,'cpu')
    assert not bad.patch_embed.proj._forward_hooks


def test_source_changed_after_import_rejected(tmp_path):
    source=tmp_path/'seed_sweep_v2.py'
    source.write_text((ROOT/'seed_sweep_v2.py').read_text())
    m=load_module(path=str(source))
    source.write_text(source.read_text()+'\n# changed after import\n')
    with pytest.raises(RuntimeError,match='changed since code was loaded'):
        m._entry_source()


def test_plain_notebook_exec_source_is_verified():
    source=(ROOT/'seed_sweep_v2.py').read_text()
    namespace={'EXP':'S1','__SRC_PATH__':str(ROOT/'seed_sweep_v2.py')}
    exec(source,namespace)
    assert namespace['_entry_source']()[1]==hashlib.sha256(source.encode()).hexdigest()
    bad={'EXP':'S1','__SRC_PATH__':str(ROOT/'seed_sweep.py')}
    exec(source,bad)
    with pytest.raises(RuntimeError,match='does not match executed run code'):
        bad['_entry_source']()


@pytest.mark.parametrize('key,value',[('weight_decay',.01),('mosaic',0.),('resume',True),('compile',True)])
def test_spatial_gate_rejects_changed_training_settings(kd,key,value):
    measured=valid_measurements(kd)
    assert kd._declared_vs_measured(measured)==[]
    measured[key]=value
    assert any(key in error for error in kd._declared_vs_measured(measured))


def test_spatial_gate_compares_resolved_args(kd):
    measured=valid_measurements(kd)
    kd._SPATIAL_RUN['resolved_train_args']={'amp':False}
    assert any('amp: resolved' in error for error in kd._declared_vs_measured(measured))
