import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from src.dataset.rain_ts_litdata import RainTimeSeriesDataset
from src.tools.optical_flow_interpolator import AnyModalityAnyFramesInterpolation, farneback_params

# --- Configuration ---

#  ============ Tests ==============


def get_dataloader():
    DATA_PATHS = [
        "data2/litdata_train_2025/litdata_interval_30/202305",
        "data2/litdata_train_2025/litdata_interval_30/202306",
        # We can add more or just use a subset for testing
    ]
    print("Initializing Dataset...")
    ds = RainTimeSeriesDataset(
        DATA_PATHS,
        n_past=2,
        n_futures=1,
    )
    # Use smaller batch size for testing
    loader = DataLoader(ds, batch_size=4, num_workers=4, shuffle=False)
    return loader


def get_dataloader_npast(n_past: int):
    DATA_PATHS = [
        "data2/litdata_train_2025/litdata_interval_30/202305",
        "data2/litdata_train_2025/litdata_interval_30/202306",
        # We can add more or just use a subset for testing
    ]
    print(f"Initializing Dataset (n_past={n_past})...")
    ds = RainTimeSeriesDataset(
        DATA_PATHS,
        n_past=n_past,
        n_futures=1,
    )
    loader = DataLoader(ds, batch_size=4, num_workers=4, shuffle=False)
    return loader


def test_radar_optical_flow():
    print("\n--- Testing Radar Optical Flow ---")
    loader = get_dataloader()

    # Get one batch
    for i, batch in enumerate(loader):
        # radar_past: [B, 2, H, W]
        radar_past = batch["radar_past"]
        if radar_past.ndim == 4:  # Expected [B, T, H, W]
            # Check if channel dim is missing or is T=2?
            # User log says: radar_past: torch.Size([4, 2, 384, 384])
            pass

        print(f"Radar Batch Shape: {radar_past.shape}")

        # Take first sample
        # Frame 0 and Frame 1
        img0 = radar_past[0, 0].numpy()  # [H, W]
        img1 = radar_past[0, 1].numpy()  # [H, W]

        # Normalize 0-1 (Dataset usually returns normalized?
        # Litdata code: radar = radar / _radar_max_value (60). So 0-1 ish.
        # But let's act robustly.
        img0_norm = np.clip(img0, 0, 1)
        img1_norm = np.clip(img1, 0, 1)

        # Interpolate
        print("Running interpolation...")
        interpolator = AnyModalityAnyFramesInterpolation("radar", interp_n_frames=1)
        interp_seq = interpolator(np.stack([img0_norm, img1_norm], axis=0))
        interp = interp_seq[1]

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img0_norm.squeeze(), cmap="jet")
        axes[0].set_title("Radar T=0")
        axes[1].imshow(interp.squeeze(), cmap="jet")
        axes[1].set_title("Radar Interpolated (T=0.5)")
        axes[2].imshow(img1_norm.squeeze(), cmap="jet")
        axes[2].set_title("Radar T=1")

        plt.savefig("radar_optical_flow_test.png")
        print("Result saved to radar_optical_flow_test.png")

        break  # Test one batch


def test_satellite_optical_flow():
    print("\n--- Testing Satellite Optical Flow ---")
    loader = get_dataloader()

    for i, batch in enumerate(loader):
        # satellite_past: [B, C, T, H, W] -> [4, 10, 2, 384, 384]
        sat_past = batch["satellite_past"]
        print(f"Satellite Batch Shape: {sat_past.shape}")

        # Take first sample
        # Frame 0 and Frame 1
        # sat_past[0] -> [10, 2, H, W]
        # Channels are dim 0.
        img0 = sat_past[0, :, 0].numpy()  # [10, H, W]
        img1 = sat_past[0, :, 1].numpy()  # [10, H, W]

        img0_norm = np.clip(img0, 0, 1)
        img1_norm = np.clip(img1, 0, 1)

        print("Running interpolation (10 channels)...")
        interpolator = AnyModalityAnyFramesInterpolation("satellite", interp_n_frames=1)
        interp_seq = interpolator(np.stack([img0_norm, img1_norm], axis=1))

        # Visualize all bands to check alignment across channels
        channels_to_show = list(range(img0_norm.shape[0]))  # all bands
        rows = len(channels_to_show)
        fig, axes = plt.subplots(rows, 3, figsize=(12, 3 * rows))

        # Ensure axes is always 2D for easy indexing
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for idx, ch in enumerate(channels_to_show):
            axes[idx, 0].imshow(interp_seq[ch, 0], cmap="gray")
            axes[idx, 0].set_title(f"Sat Ch{ch} T=0")

            axes[idx, 1].imshow(interp_seq[ch, 1], cmap="gray")
            axes[idx, 1].set_title(f"Sat Ch{ch} Interp")

            axes[idx, 2].imshow(interp_seq[ch, 2], cmap="gray")
            axes[idx, 2].set_title(f"Sat Ch{ch} T=1")

            for col in range(3):
                axes[idx, col].axis("off")

        plt.tight_layout()
        plt.savefig("satellite_optical_flow_test.png")
        print("Result saved to satellite_optical_flow_test.png")

        break


def test_radar_interp_0_5():
    print("\n--- Testing Radar Interp Frames 0->5 (insert 10) ---")
    loader = get_dataloader_npast(6)

    for _, batch in enumerate(loader):
        radar_past = batch["radar_past"]  # [B, T, H, W]
        print(f"Radar Batch Shape: {radar_past.shape}")

        frame0 = radar_past[0, 0].numpy()
        frame5 = radar_past[0, 5].numpy()

        frame0 = np.clip(frame0, 0, 1)
        frame5 = np.clip(frame5, 0, 1)

        interpolator = AnyModalityAnyFramesInterpolation("radar", interp_n_frames=10)
        seq = interpolator(np.stack([frame0, frame5], axis=0))  # [12, H, W]

        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.ravel()
        for idx in range(seq.shape[0]):
            axes[idx].imshow(seq[idx], cmap="jet")
            axes[idx].set_title(f"T{idx}")
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig("radar_interp_0_5_insert10.png")
        print("Result saved to radar_interp_0_5_insert10.png")
        break


def test_satellite_feature5_interp_0_5():
    print("\n--- Testing Satellite Feature 5 Interp Frames 0->5 (insert 10) ---")
    loader = get_dataloader_npast(6)

    for _, batch in enumerate(loader):
        sat_past = batch["satellite_past"]  # [B, C, T, H, W]
        print(f"Satellite Batch Shape: {sat_past.shape}")

        frame0 = sat_past[0, 5, 0].numpy()
        frame5 = sat_past[0, 5, 5].numpy()

        frame0 = np.clip(frame0, 0, 1)
        frame5 = np.clip(frame5, 0, 1)

        interpolator = AnyModalityAnyFramesInterpolation("satellite", interp_n_frames=10)
        seq = interpolator(np.stack([frame0, frame5], axis=0)[np.newaxis, ...])  # [1, 12, H, W]
        seq = seq[0]

        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        axes = axes.ravel()
        for idx in range(seq.shape[0]):
            axes[idx].imshow(seq[idx], cmap="gray")
            axes[idx].set_title(f"T{idx}")
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig("satellite_ch5_interp_0_5_insert10.png")
        print("Result saved to satellite_ch5_interp_0_5_insert10.png")
        break


if __name__ == "__main__":
    # test_radar_optical_flow()
    # test_satellite_optical_flow()
    test_radar_interp_0_5()
    test_satellite_feature5_interp_0_5()
