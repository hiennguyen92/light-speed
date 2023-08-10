import torch  # isort:skip
import json
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import commons
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import MultiPeriodDiscriminator, SynthesizerTrn
from tfloader import load_tfdata

tf.config.set_visible_devices([], "GPU")

parser = ArgumentParser()
parser.add_argument("--config", type=str, default="config.json")
parser.add_argument("--tfdata", type=str, default="data/tfdata")
parser.add_argument("--log-dir", type=Path, default="logs")
parser.add_argument("--ckpt-dir", type=Path, default="logs")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--compile", action="store_true", default=False)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ckpt-interval", type=int, default=5_000)
FLAGS = parser.parse_args()

# credit: https://github.com/karpathy/nanoGPT/blob/master/train.py#L72-L112
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed(FLAGS.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = FLAGS.device
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = FLAGS.compile
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)
# initialize a GradScaler. If enabled=False scaler is a no-op
print(dtype, ptdtype, ctx)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
FLAGS.ckpt_dir.mkdir(exist_ok=True, parents=True)

with open(FLAGS.config, "rb") as f:
    hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
torch.manual_seed(hps.train.seed)

train_writer = SummaryWriter(FLAGS.log_dir / "train", flush_secs=30)
val_writer = SummaryWriter(FLAGS.log_dir / "val", flush_secs=30)

net_g = SynthesizerTrn(
    256,
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **vars(hps.model),
).to(device)
net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
optim_g = torch.optim.AdamW(
    net_g.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
)
optim_d = torch.optim.AdamW(
    net_d.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
)

if compile:
    # net_g = torch.compile(net_g)
    net_d = torch.compile(net_d)


scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)

net_g.train()
net_d.train()

ds, num_batch_per_epoch = load_tfdata(FLAGS.tfdata, "train", FLAGS.batch_size, epoch)
step = 0
for batch in tqdm(ds.as_numpy_iterator()):
    step = step + 1
    x = torch.from_numpy(batch["phone_idx"]).to(device, non_blocking=True).long()
    x_lengths = (
        torch.from_numpy(batch["phone_length"]).to(device, non_blocking=True).long()
    )
    spec = (
        torch.from_numpy(batch["spec"])
        .to(device, non_blocking=True)
        .swapaxes(-1, -2)
        .float()
    )
    spec_lengths = (
        torch.from_numpy(batch["spec_length"]).to(device, non_blocking=True).long()
    )
    y = torch.from_numpy(batch["wav"]).to(device, non_blocking=True).float()[:, None, :]
    y_lengths = (
        torch.from_numpy(batch["wav_length"]).to(device, non_blocking=True).long()
    )
    duration = (
        torch.from_numpy(batch["phone_duration"]).to(device, non_blocking=True).float()
    )
    end_time = torch.cumsum(duration, dim=-1)
    start_time = end_time - duration
    start_frame = (
        start_time * hps.data.sampling_rate / hps.data.hop_length / 1000
    ).int()
    end_frame = (end_time * hps.data.sampling_rate / hps.data.hop_length / 1000).int()
    pos = torch.arange(0, spec.shape[-1], device=spec.device)
    attn = torch.logical_and(
        pos[None, :, None] >= start_frame[:, None, :],
        pos[None, :, None] < end_frame[:, None, :],
    )

    with ctx:
        (
            y_hat,
            l_length,
            attn,
            ids_slice,
            x_mask,
            z_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        ) = net_g(x, x_lengths, attn.float(), spec, spec_lengths)

    mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
    )
    y_mel = commons.slice_segments(
        mel, ids_slice, hps.train.segment_size // hps.data.hop_length
    )
    y_hat_mel = mel_spectrogram_torch(
        y_hat.float().squeeze(1),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
    )

    y = commons.slice_segments(
        y, ids_slice * hps.data.hop_length, hps.train.segment_size
    )  # slice

    with ctx:
        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
    loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with ctx:
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)

    loss_mel = F.l1_loss(y_mel, y_hat_mel)
    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)

    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, losses_gen = generator_loss(y_d_hat_g)
    loss_gen_all = (
        loss_gen + loss_fm + loss_mel * hps.train.c_mel + loss_kl * hps.train.c_kl
    )

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    train_writer.add_scalar("loss_disc_all", loss_disc_all.float(), global_step=step)
    train_writer.add_scalar("loss_gen_all", loss_gen_all.float(), global_step=step)
    train_writer.add_scalar("loss_gen", loss_gen.float(), global_step=step)
    train_writer.add_scalar("loss_fm", loss_fm.float(), global_step=step)
    train_writer.add_scalar("loss_mel", loss_mel.float(), global_step=step)
    train_writer.add_scalar("loss_kl", loss_kl.float(), global_step=step)
    train_writer.add_scalar("grad_scale", scaler.get_scale(), global_step=step)
    train_writer.add_scalar("grad_norm_d", grad_norm_d, global_step=step)
    train_writer.add_scalar("grad_norm_g", grad_norm_g, global_step=step)
    if step % FLAGS.ckpt_interval == 0:
        torch.save(
            {
                "step": step,
                "net_g": net_g.state_dict(),
                "net_d": net_d.state_dict(),
                "scaler": scaler.state_dict(),
                "optim_d": optim_d.state_dict(),
                "optim_g": optim_g.state_dict(),
                "scheduler_g": scheduler_g.state_dict(),
                "scheduler_d": scheduler_d.state_dict(),
            },
            FLAGS.ckpt_dir / f"ckpt_{step:08d}.pth",
        )
    if step % num_batch_per_epoch == 0:
        lr = optim_g.param_groups[0]["lr"]
        train_writer.add_scalar("lr", lr, global_step=step)
        scheduler_g.step()
        scheduler_d.step()
