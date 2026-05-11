import io
import os
import time
from datetime import datetime

import numpy as np
import psutil
import torch
import webdataset as wds

IMAGE_H, IMAGE_W = 256, 256

# 缓存时间编码，避免重复计算
_sin_cache = {}
_cos_cache = {}


def _generate_time_encoding(time_obj: datetime) -> torch.Tensor:
    key = (time_obj.hour, time_obj.minute)
    if key in _sin_cache:
        return _sin_cache[key]

    minutes_in_day = 24 * 60
    time_of_day_minute = time_obj.hour * 60 + time_obj.minute
    angle = (time_of_day_minute / minutes_in_day) * (2 * np.pi)
    sin_emb = np.sin(angle)
    cos_emb = np.cos(angle)

    sin_mat = np.full((IMAGE_H, IMAGE_W), sin_emb, dtype=np.float32)
    cos_mat = np.full((IMAGE_H, IMAGE_W), cos_emb, dtype=np.float32)

    time_encoding = np.stack([sin_mat, cos_mat], axis=0)
    time_encoding_tensor = torch.from_numpy(time_encoding)

    _sin_cache[key] = time_encoding_tensor
    return time_encoding_tensor


def npy_decoder(key, data):
    if key.endswith(".npy"):
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        return arr
    else:
        return data


def npy_decode_io(npy_bytes: bytes) -> np.ndarray:
    return np.load(io.BytesIO(npy_bytes))


def webdataset_sample_transform(sample):
    sample_key = sample["__key__"]
    time_obj = datetime.strptime(sample_key, "%Y%m%d_%H%M")
    time_encoding_tensor = _generate_time_encoding(time_obj)

    sample["time_enc"] = time_encoding_tensor
    return sample


def remove_meta_data(sample):
    _keys = sample.keys()
    for k in _keys:
        if k.startswith("__"):
            del sample[k]


def remove_extension(sample):
    new_sample = {}
    for k, v in sample.items():
        k: str
        if k.startswith("__"):
            new_sample[k] = v
        elif len(name_ext := k.rsplit(".", 1)) > 1:
            name, ext = name_ext
            new_sample[name] = v
        else:
            new_sample[k] = v

    return new_sample

def is_valid_sample(sample):
    required_keys = [f"radar{i}" for i in range(6)] + \
                    [f"satellite{i}" for i in range(4)] + ["rainfall"]
    return all(k in sample for k in required_keys)


def get_webdataset_dataloader(tar_file_urls, batch_size, num_workers, shuffle_size=5000):
    dataset = wds.WebDataset(
        tar_file_urls,
        handler=wds.warn_and_continue,
        resampled=False,
        shardshuffle=False,
        seed=42,
    )

    dataset = dataset.decode("torch")
    if shuffle_size > 0:
        dataset = dataset.shuffle(shuffle_size)

    # !!! 这个时间encoding参考diffusion Unet的写法去写
    # dataset = dataset.map(webdataset_sample_transform)

    dataset = dataset.map(remove_extension)

    dataset = dataset.select(is_valid_sample)

    dataloader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    # TAR_DATA_ROOT = "/HardDisk/JieYiZhu/MMRainPrediction/data/pair_shards_30min"
    # tar_list_filepath = os.path.join(TAR_DATA_ROOT, TAR_FILE_LIST_NAME)

    # if not os.path.exists(tar_list_filepath):
    #     print(f"错误: 找不到 {tar_list_filepath}")
    #     exit(1)

    # with open(tar_list_filepath, "r") as f:
    #     relative_tar_urls = [line.strip() for line in f if line.strip()]
    # all_tar_urls = [os.path.join(TAR_DATA_ROOT, url) for url in relative_tar_urls]

    BATCH_SIZE = 64
    NUM_WORKERS = 1

    from pathlib import Path

    # tar_s = [p.as_posix() for p in Path("/Data2/JieYiZhu/Multimodal-Rain-Prediction/data/pair_shards_30min").glob("*.tar")]
    tar_s = "/Data2/JieYiZhu/Multimodal-Rain-Prediction/data/rainfall_tar_30min/sample_000005.tar"

    loader = get_webdataset_dataloader(tar_s, BATCH_SIZE, NUM_WORKERS, shuffle_size=512)

    print("开始加载数据...")
    t_start_batch = time.perf_counter()
    for i, sample in enumerate(loader):
        t_end_batch = time.perf_counter()
        print(sample['__key__'])
        print(
            f"第 {i} 个 batch 加载完成，耗时: {t_end_batch - t_start_batch:.4f} 秒, Key: {sample.keys()}"
        )
        t_start_batch = time.perf_counter()
        if i > 1:
            break
