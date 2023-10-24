import json
from pathlib import Path
import numpy as np
import torch
import json
import librosa
import tensorflow as tf
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser

mel_basis = {}
hann_window = {}

parser = ArgumentParser()
parser.add_argument("--dataset-dir", type=Path, default="dataset")
FLAGS = parser.parse_args()

def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    global hann_window
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = spec.squeeze(0)
    return torch.swapaxes(spec, 0, 1)


def tensor_to_bytes(t):
    t = tf.constant(t)
    t = tf.io.serialize_tensor(t)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[t.numpy()]))


def write_tfdata(data, out_file):
    with tf.io.TFRecordWriter(out_file) as file_writer:
        for wav_file, seq, _ in data:
            phone_seq = []
            for phone, duration in seq:
                phone_idx = phone_set.index(phone)
                phone_seq.append((phone_idx, duration))
            phone_seq = np.array(phone_seq, dtype=np.float32)

            # load wav
            wav, sr = librosa.load(wav_file, sr=config["data"]["sampling_rate"], dtype=np.float32)
            wav = torch.from_numpy(wav)
            # compute spec
            spec = spectrogram_torch(
                wav[None],
                n_fft=config["data"]["filter_length"],
                sampling_rate=config["data"]["sampling_rate"],
                hop_size=config["data"]["hop_length"],
                win_size=config["data"]["win_length"],
                center=False
            )
            features = {
                "phone_idx": tensor_to_bytes(phone_seq[:, 0].astype(np.int32)),
                "phone_duration": tensor_to_bytes(phone_seq[:, 1]),
                "wav": tensor_to_bytes(wav.half().numpy()),
                "spec": tensor_to_bytes(spec.half().numpy())
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            file_writer.write(example.SerializeToString())

def write_split(split, data, num_chunks):
    data = np.array(data, dtype=object)
    chunks = list(np.array_split(data, num_chunks))
    for i, chunk in enumerate(tqdm(chunks)):
        write_tfdata(chunk, f"tfdata/{split}/part_{i:03d}.tfrecords")

with open("../config.json", "rb") as f:
    config = json.load(f)
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = FLAGS.dataset_dir
json_files = sorted(data_dir.glob("*.json"))
dataset = []
phone_set = []

for file_path in json_files:
    with open(file_path, "rb") as f:
        data = json.load(f)
    seq = []
    word_index = 0
    words =  data["tiers"]["words"]["entries"]
    for start, end, phone in data["tiers"]["phones"]["entries"]:
        if start > words[word_index][1] - 1e-5:
            seq.append( ("<SEP>", 0) )
            word_index += 1
        duration = end * 1000 - start * 1000 # ms
        phone_set.append(phone)
        seq.append( (phone, duration) )
    wav_file = file_path.with_suffix(".mp3")
    dataset.append((wav_file, seq, data["end"]))

phone_set = ["<SEP>"] + sorted(set(phone_set))
assert len(phone_set) <= 256
with open("phone_set.json", "w", encoding="utf-8") as f:
    json.dump(phone_set, f)

assert phone_set.index("<SEP>") == 0

random.Random(42).shuffle(dataset)
L = len(dataset) - 256
train_data = dataset[:L]
test_data = dataset[L:]
print("Train data size:", len(train_data))
print("Test data size:", len(test_data))

write_split("test", test_data, 1)
write_split("train", train_data, 256)
