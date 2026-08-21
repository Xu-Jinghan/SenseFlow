import csv
import json
from pathlib import Path

import numpy as np


class CenterPixelWaveformRecorder:
    def __init__(self, enabled, output_dir, target_channel=1, max_plot_frames=120, title="Center pixel waveform"):
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.target_channel = int(target_channel)
        self.max_plot_frames = int(max_plot_frames)
        self.title = str(title)
        self.rows = []
        self.frame_rows = []
        self.global_step = 0
        self.center_xy = None

    def _extract_center_power(self, power_maps):
        arr = np.asarray(power_maps, dtype=np.float64)
        if arr.ndim == 2:
            center_y = arr.shape[0] // 2
            center_x = arr.shape[1] // 2
            self.center_xy = [int(center_x), int(center_y)]
            return np.asarray([arr[center_y, center_x]], dtype=np.float64)
        if arr.ndim == 3:
            center_y = arr.shape[1] // 2
            center_x = arr.shape[2] // 2
            self.center_xy = [int(center_x), int(center_y)]
            return np.asarray(arr[:, center_y, center_x], dtype=np.float64)
        raise ValueError(f"Unsupported power map rank for center-pixel trace: {arr.ndim}")

    def record(self, frame_offset, frame_name, power_maps, center_trace, dt_s, video_fps):
        if not self.enabled:
            return
        if center_trace is None:
            raise ValueError("center_trace is required when center waveform recording is enabled")

        center_power = self._extract_center_power(power_maps)
        center_iout = np.asarray(center_trace, dtype=np.float64)
        if center_iout.ndim == 1:
            center_iout = center_iout[:, None]

        channel_count = center_iout.shape[1]
        target_channel = int(np.clip(self.target_channel, 0, max(0, channel_count - 1)))
        target_power_channel = int(np.clip(target_channel, 0, max(0, center_power.size - 1)))
        frame_time_s = float(frame_offset) / float(video_fps)

        frame_row = {
            "frame_offset": int(frame_offset),
            "frame_name": str(frame_name),
            "frame_time_s": frame_time_s,
            "target_channel": target_channel,
            "power_w_target": float(center_power[target_power_channel]),
            "iout_a_target_last": float(center_iout[-1, target_channel]),
        }
        for channel_idx in range(3):
            power_value = center_power[channel_idx] if channel_idx < center_power.size else np.nan
            iout_value = center_iout[-1, channel_idx] if channel_idx < channel_count else np.nan
            frame_row[f"power_w_ch{channel_idx}"] = float(power_value)
            frame_row[f"iout_a_ch{channel_idx}_last"] = float(iout_value)
        self.frame_rows.append(frame_row)

        for step_idx in range(center_iout.shape[0]):
            row = {
                "global_step": int(self.global_step),
                "frame_offset": int(frame_offset),
                "frame_name": str(frame_name),
                "step_in_frame": int(step_idx),
                "time_s": float(self.global_step) * float(dt_s),
                "target_channel": target_channel,
                "power_w_target": float(center_power[target_power_channel]),
                "iout_a_target": float(center_iout[step_idx, target_channel]),
            }
            for channel_idx in range(3):
                power_value = center_power[channel_idx] if channel_idx < center_power.size else np.nan
                iout_value = center_iout[step_idx, channel_idx] if channel_idx < channel_count else np.nan
                row[f"power_w_ch{channel_idx}"] = float(power_value)
                row[f"iout_a_ch{channel_idx}"] = float(iout_value)
            self.rows.append(row)
            self.global_step += 1

    def finalize(self, sensor_args, base_params):
        if not self.enabled:
            return None
        if not self.rows:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        waveform_csv = self.output_dir / "center_pixel_waveform.csv"
        frame_csv = self.output_dir / "center_pixel_frame_summary.csv"
        waveform_png = self.output_dir / "center_pixel_waveform.png"
        summary_path = self.output_dir / "center_pixel_waveform_summary.json"

        self._write_csv(waveform_csv, self.rows)
        self._write_csv(frame_csv, self.frame_rows)
        self._save_plot(waveform_png)

        power_target = np.asarray([row["power_w_target"] for row in self.rows], dtype=np.float64)
        iout_target = np.asarray([row["iout_a_target"] for row in self.rows], dtype=np.float64)
        summary = {
            "title": self.title,
            "num_frames": int(len(self.frame_rows)),
            "num_steps": int(len(self.rows)),
            "video_fps": float(getattr(sensor_args, "video_fps")),
            "fps_sim": float(getattr(sensor_args, "fps_sim")),
            "readout": str(getattr(sensor_args, "readout")),
            "analog_readout": str(getattr(sensor_args, "analog_readout")),
            "shot_noise": int(getattr(sensor_args, "shot_noise", 0)),
            "use_noise_fn": int(getattr(sensor_args, "use_noise_fn", 0)),
            "spatial_variation_r_pct": float(base_params.get("spatial_variation_r_pct", 0.0)),
            "spatial_variation_r_ratio": float(base_params.get("spatial_variation_r_ratio", 0.0)),
            "params_csv": str(getattr(sensor_args, "params_csv")),
            "center_xy": self.center_xy,
            "target_channel": int(self.frame_rows[0]["target_channel"]),
            "power_w_target_min": float(np.min(power_target)),
            "power_w_target_max": float(np.max(power_target)),
            "iout_a_target_min": float(np.min(iout_target)),
            "iout_a_target_max": float(np.max(iout_target)),
            "outputs": {
                "waveform_csv": str(waveform_csv),
                "frame_summary_csv": str(frame_csv),
                "waveform_png": str(waveform_png),
                "summary_json": str(summary_path),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    @staticmethod
    def _write_csv(path, rows):
        fieldnames = list(rows[0].keys())
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _save_plot(self, path):
        if not self.rows:
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames_to_plot = min(len(self.frame_rows), max(1, self.max_plot_frames))
        rows_to_plot = [
            row
            for row in self.rows
            if int(row["frame_offset"]) < frames_to_plot
        ]
        time_axis = np.asarray([row["time_s"] for row in rows_to_plot], dtype=np.float64)
        power_axis = np.asarray([row["power_w_target"] for row in rows_to_plot], dtype=np.float64)
        iout_axis = np.asarray([row["iout_a_target"] for row in rows_to_plot], dtype=np.float64)

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].plot(time_axis, power_axis, lw=1.2)
        axes[0].set_ylabel("P center (W)")
        axes[0].set_title(self.title)
        axes[1].plot(time_axis, iout_axis, lw=1.2, color="#b03030")
        axes[1].set_ylabel("Iout center (A)")
        axes[1].set_xlabel("time (s)")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
