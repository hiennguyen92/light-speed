import torch  # isort:skip
from tqdm.cli import tqdm
from transformers import Adafactor
from torch_ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import tensorflow as tf
from argparse import ArgumentParser
from pathlib import Path
from models import DurationNet
device = "cuda" if torch.cuda.is_available() else "cpu"
tf.config.set_visible_devices([], 'GPU')

parser = ArgumentParser()
parser.add_argument("--ckpt-dir", type=Path, default="ckpts")
parser.add_argument("--tfdata", type=str, default="data/tfdata")
FLAGS = parser.parse_args()


def load_tfdata(root, split, batch_size):
    feature_description = {
        "phone_idx": tf.io.FixedLenFeature([], tf.string),
        "phone_duration": tf.io.FixedLenFeature([], tf.string),
    }

    def parse_tfrecord(r):
        r = tf.io.parse_example(r, feature_description)
        phone_idx = tf.reshape(tf.io.parse_tensor(r["phone_idx"], out_type=tf.int32), [-1])
        phone_duration = tf.reshape(
            tf.io.parse_tensor(r["phone_duration"], out_type=tf.float32), [-1]
        )
        return {
            "phone_idx": phone_idx,
            "phone_duration": phone_duration,
            "phone_length": tf.shape(phone_duration)[0],
        }

    files = tf.data.Dataset.list_files(f"{root}/{split}/part_*.tfrecords")
    return (
        tf.data.TFRecordDataset(files, num_parallel_reads=4)
        .map(parse_tfrecord, num_parallel_calls=4)
        .shuffle(buffer_size=batch_size * 32)
        .bucket_by_sequence_length(
            lambda x: x["phone_length"],
            bucket_boundaries=(32, 64, 128, 256, 512),
            bucket_batch_sizes=[batch_size] * 6,
            pad_to_bucket_boundary=False,
            drop_remainder=True,
        )
        .prefetch(1)
    )


net = DurationNet(256, 64, 4).to(device)
optim = Adafactor(net.parameters(), warmup_init=False)
ema = ExponentialMovingAverage(net.parameters(), decay=0.995)
batch_size = 4
num_epochs = 50

def loss_fn(net, batch):
    token = batch["phone_idx"]
    duration = batch["phone_duration"] / 1000.
    length = batch["phone_length"]
    mask = torch.arange(0, duration.shape[1], device=device)[None, :] < length[:, None]
    y = net(token, length).squeeze(-1)
    loss = torch.nn.functional.l1_loss(y, duration, reduction="none")
    loss = torch.where(mask == 1, loss, 0.0)
    loss = torch.sum(loss) / torch.sum(mask)
    return loss


ds = load_tfdata(FLAGS.tfdata, "train", batch_size)
val_ds = load_tfdata(FLAGS.tfdata, "test", batch_size)

def prepare_batch(batch):
    return {
        "phone_idx": torch.from_numpy(batch["phone_idx"]).to(device, non_blocking=True),
        "phone_duration": torch.from_numpy(batch["phone_duration"]).to(device, non_blocking=True),
        "phone_length": torch.from_numpy(batch["phone_length"]).to(device, non_blocking=True),
    }

for epoch in range(num_epochs):
    losses = []
    for batch in tqdm(ds.as_numpy_iterator()):
        batch = prepare_batch(batch)
        loss = loss_fn(net, batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        ema.update()
        losses.append(loss.item())
    train_loss = sum(losses) / len(losses)
    
    losses = []
    with ema.average_parameters():    
        with torch.inference_mode():
            net.eval()
            for batch in val_ds.as_numpy_iterator():
                batch = prepare_batch(batch)
                loss = loss_fn(net, batch)
                losses.append(loss.item())
            net.train()
    val_loss = sum(losses) / len(losses)
    print(f"epoch {epoch:<3d}  train loss {train_loss:.5}  val loss {val_loss:.5f}")


ema.copy_to(net.parameters())
net = net.eval()
torch.save(net.state_dict(), FLAGS.ckpt_dir / f"duration_model.pth")

with torch.inference_mode():
    for batch in val_ds.as_numpy_iterator():
        batch = prepare_batch(batch)
        duration = batch["phone_duration"] / 1000
        y = net(batch["phone_idx"], batch["phone_length"]).squeeze(-1)
        break
    
plt.figure(figsize=(10, 5))
d = duration[0].tolist()
t = y[0].tolist()
plt.plot(d, '-*', label="target")
plt.plot(t, '-*', label="predict")
plt.legend()
