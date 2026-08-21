"""
Video-level photodetector non-ideality simulation.

Pipeline: video frames -> 128x128 optical-power array -> pixel-level
non-ideal-effect simulation -> three readout modes -> output videos.

Three readout modes:
  1. TIA: instantaneous current readout; map I_out directly to pixel values
  2. Integration: integrate current during exposure, Q = ∫I·dt, to emulate
     CMOS integrating readout
  3. ADC: quantize the integrated result to N bits to emulate real ADC
     conversion

Usage:
  python video_simulate.py [--fps 30] [--adc_bits 8] [--P_max 1] ...
"""

import numpy as np
import cv2
import argparse
import math
from pathlib import Path

from photodetector_array import (
    PhotodetectorArray,
    ReadoutADC,
    ReadoutIntegration,
    ReadoutTIA,
    simulate_video,
)
from photodetector_model import NONLINEAR_POWER_REF_W


# ============================================================
# 3. Video processing pipeline
# ============================================================

def load_video_as_power(video_path, target_size=(128, 128), P_max=1.0,
                        max_output_frames=None, fps_output=None):
    """
    Load a video, convert it to grayscale, and map it into optical-power arrays.

    The grayscale range [0, 255] is mapped linearly onto
    [NONLINEAR_POWER_REF_W, P_max].

    Returns
    -------
    frames: list of (H, W) float64 arrays, optical power (W)
    fps_original: original frame rate
    """
    if P_max < NONLINEAR_POWER_REF_W:
        raise ValueError(
            f"P_max must be >= NONLINEAR_POWER_REF_W ({NONLINEAR_POWER_REF_W:.3e} W), got {P_max:.3e} W"
        )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    max_input_frames = None

    if max_output_frames is not None:
        if fps_output is None or fps_output <= 0:
            raise ValueError("fps_output must be positive when max_output_frames is set")
        max_duration = max_output_frames / fps_output
        max_input_frames = max(2, int(math.ceil(max_duration * fps)) + 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (target_size[1], target_size[0]),
                             interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float64) / 255.0
        P = NONLINEAR_POWER_REF_W + normalized * (P_max - NONLINEAR_POWER_REF_W)
        frames.append(P)
        if max_input_frames is not None and len(frames) >= max_input_frames:
            break

    cap.release()
    limit_msg = ""
    if max_input_frames is not None:
        limit_msg = f", 截取前 {len(frames)} 帧输入"
    print(f"  读取 {len(frames)} 帧, 原始 fps={fps:.1f}{limit_msg}, "
          f"缩放到 {target_size[0]}×{target_size[1]}, "
          f"功率映射范围 {NONLINEAR_POWER_REF_W:.3e} ~ {P_max:.3e} W")
    return frames, fps


def interpolate_frames(frames, fps_original, fps_sim):
    """
    Temporal interpolation from the original frame rate to the simulation-step rate.
    Use linear interpolation to generate sub-frames between frames.
    """
    n_orig = len(frames)
    T_total = (n_orig - 1) / fps_original
    n_sim = int(T_total * fps_sim) + 1
    t_orig = np.arange(n_orig) / fps_original
    t_sim = np.linspace(0, T_total, n_sim)

    H, W = frames[0].shape
    frames_stack = np.stack(frames, axis=0)  # (n_orig, H, W)

    # Pixel-wise linear interpolation
    interp_frames = []
    for i in range(n_sim):
        t = t_sim[i]
        idx = t * fps_original
        idx_lo = min(int(np.floor(idx)), n_orig - 1)
        idx_hi = min(idx_lo + 1, n_orig - 1)
        frac = idx - idx_lo
        frame = frames_stack[idx_lo] * (1 - frac) + frames_stack[idx_hi] * frac
        interp_frames.append(frame)

    return interp_frames, t_sim


def frames_to_video(frames, output_path, fps):
    """Write floating-point frame arrays to a grayscale video."""
    if not frames:
        print("  无输出帧!")
        return

    H, W = frames[0].shape

    # Global normalization
    all_vals = np.concatenate([f.ravel() for f in frames])
    vmin = np.percentile(all_vals, 1)
    vmax = np.percentile(all_vals, 99)
    if vmax <= vmin:
        vmax = vmin + 1e-10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_size = 512
    writer = cv2.VideoWriter(str(output_path), fourcc, fps,
                             (out_size, out_size))

    for frame in frames:
        normalized = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
        img8 = (normalized * 255).astype(np.uint8)
        upscaled = cv2.resize(img8, (out_size, out_size),
                              interpolation=cv2.INTER_NEAREST)
        bgr = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
        writer.write(bgr)

    writer.release()
    print(f"  保存: {output_path} ({len(frames)} 帧, {fps} fps, {out_size}×{out_size})")


# ============================================================
# 4. Comparison visualization: tile ideal output vs. the three readout modes
# ============================================================

def make_comparison_video(ideal_frames, tia_frames, integ_frames, adc_frames,
                          output_path, fps):
    """
    2x2 tiled layout: ideal | TIA | integration | ADC.
    """
    tile_size = 256
    H_out, W_out = tile_size * 2, tile_size * 2

    # Normalize each mode with its own global range because the physical units differ.
    all_sets = [ideal_frames, tia_frames, integ_frames, adc_frames]
    ranges = []
    for fset in all_sets:
        vals = np.concatenate([f.ravel() for f in fset])
        vmin = np.percentile(vals, 1)
        vmax = np.percentile(vals, 99)
        if vmax <= vmin:
            vmax = vmin + 1e-10
        ranges.append((vmin, vmax))

    def normalize_frame(frame, idx):
        vmin, vmax = ranges[idx]
        n = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
        return (n * 255).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps,
                             (W_out, H_out))

    n_frames = min(len(ideal_frames), len(tia_frames),
                   len(integ_frames), len(adc_frames))

    labels = ["Ideal", "TIA", "Integration", f"ADC"]
    positions = [(10, 25), (tile_size + 10, 25),
                 (10, tile_size + 25), (tile_size + 10, tile_size + 25)]

    for i in range(n_frames):
        tiles = [
            normalize_frame(ideal_frames[i], 0),
            normalize_frame(tia_frames[i], 1),
            normalize_frame(integ_frames[i], 2),
            normalize_frame(adc_frames[i], 3),
        ]

        canvas = np.zeros((H_out, W_out, 3), dtype=np.uint8)

        for j, (tile, label, pos) in enumerate(zip(tiles, labels, positions)):
            r, c = divmod(j, 2)
            y0, x0 = r * tile_size, c * tile_size
            resized = cv2.resize(tile, (tile_size, tile_size),
                                 interpolation=cv2.INTER_NEAREST)
            gray_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            canvas[y0:y0+tile_size, x0:x0+tile_size] = gray_bgr

            # Label
            cv2.putText(canvas, label, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(canvas)

    writer.release()
    print(f"  保存对比视频: {output_path} ({n_frames} 帧, 2×2)")


def _format_scalar(value, scale=1.0, unit=""):
    suffix = f" {unit}" if unit else ""
    return f"{value * scale:.3e}{suffix}"


def _format_stats(stats, scale=1.0, unit=""):
    suffix = f" {unit}" if unit else ""
    return (
        f"mean={stats['mean'] * scale:.3e}, std={stats['std'] * scale:.3e}, "
        f"min={stats['min'] * scale:.3e}, max={stats['max'] * scale:.3e}{suffix}"
    )


def print_nonideal_parameter_summary(array, args, fps_orig, dt_sim):
    summary = array.summarize_nonideal_effects()
    params = summary["shared_params"]
    noise = summary["shared_noise"]
    variation = summary["variation"]
    t_int = 1.0 / args.fps_output

    print("\n" + "=" * 60)
    print("Step 2.5: 当前视频仿真的非理想参数")
    print(f"  阵列尺寸 = {summary['shape'][0]}x{summary['shape'][1]}, rng_seed = {summary['rng_seed']}")
    print(f"  输入视频帧率 = {fps_orig:.3f} Hz")
    print(f"  仿真步进 = {args.fps_sim:.3f} Hz, dt_sim = {dt_sim * 1e3:.3f} ms")
    print(f"  输出帧率 = {args.fps_output:.3f} Hz, 积分时长 = {t_int * 1e3:.3f} ms")
    print(
        "  输入光功率映射 = "
        f"{_format_scalar(NONLINEAR_POWER_REF_W, unit='W')} ~ "
        f"{_format_scalar(args.P_max, unit='W')}"
    )
    print(f"  ADC 量化 = {args.adc_bits} bit")

    print("  标称器件参数:")
    print(
        "    fast carrier: "
        f"R_fast={_format_scalar(params['R_fast'], unit='A/W')}, "
        f"eta_fast={params['eta_fast']:.3f}, "
        f"tau_rise_fast={_format_scalar(params['tau_rise_fast'], 1e3, 'ms')}, "
        f"tau_fall_fast={_format_scalar(params['tau_fall_fast'], 1e3, 'ms')}"
    )
    print(
        "    slow carrier: "
        f"R_slow={_format_scalar(params['R_slow'], unit='A/W')}, "
        f"eta_slow={params['eta_slow']:.3f}, "
        f"tau_rise_slow={_format_scalar(params['tau_rise_slow'], 1e3, 'ms')}, "
        f"tau_fall_slow={_format_scalar(params['tau_fall_slow'], 1e3, 'ms')}"
    )
    print(
        "    trap/drift: "
        f"alpha={_format_scalar(params['alpha'])}, "
        f"beta={_format_scalar(params['beta'])}, "
        f"delta={_format_scalar(params['delta'])}, "
        f"gamma={_format_scalar(params['gamma'])}, "
        f"tau_drift={_format_scalar(params['tau_drift'], 1e3, 'ms')}, "
        f"drift_scale={params['drift_scale']:.3f}"
    )

    print("  噪声与暗电流参数:")
    print(
        f"    dark_current_base = {_format_scalar(summary['dark_current_base'], 1e9, 'nA')}, "
        f"i_thermal = {_format_scalar(noise['i_thermal'], 1e9, 'nA/sqrt(Hz)')}, "
        f"bandwidth = {_format_scalar(noise['bandwidth'], unit='Hz')}, "
        f"shot_noise = {noise['shot_noise']}"
    )
    print(f"    noise_disabled = {args.disable_noise}")

    print("  像素间 variation 配置:")
    print(
        f"    responsivity_cv = {variation['responsivity_cv']:.3f}, "
        f"eta_sigma = {variation['eta_sigma']:.3f}, "
        f"tau_cv = {variation['tau_cv']:.3f}"
    )
    print(
        f"    dark_current_cv = {variation['dark_current_cv']:.3f}, "
        f"thermal_noise_cv = {variation['thermal_noise_cv']:.3f}"
    )

    print("  当前采样阵列的实际参数统计:")
    for carrier in summary["carriers"]:
        label = carrier["label"]
        print(
            f"    {label}.R (A/W): {_format_stats(carrier['R'])}"
        )
        print(
            f"    {label}.eta: {_format_stats(carrier['eta'])}"
        )
        print(
            f"    {label}.tau_rise (ms): {_format_stats(carrier['tau_rise'], 1e3)}"
        )
        print(
            f"    {label}.tau_fall (ms): {_format_stats(carrier['tau_fall'], 1e3)}"
        )
    print(f"    delta: {_format_stats(summary['delta'])}")
    print(f"    gamma: {_format_stats(summary['gamma'])}")
    print(f"    tau_drift (ms): {_format_stats(summary['tau_drift'], 1e3)}")
    print(f"    dark_current (nA): {_format_stats(summary['dark_current'], 1e9)}")
    print(f"    i_thermal (nA/sqrt(Hz)): {_format_stats(summary['i_thermal'], 1e9)}")
    print("  注: TIA / Integration / ADC 三路使用相同的标称参数与随机种子配置。")


# ============================================================
# 5. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Photodetector array video simulation")
    parser.add_argument("--video", default="test_video/IMG_7489 2.mov")
    parser.add_argument("--size", type=int, default=128, help="像素阵列边长")
    parser.add_argument("--fps_output", type=float, default=10, help="输出帧率")
    parser.add_argument("--fps_sim", type=float, default=1000, help="仿真步进帧率 (Hz)")
    parser.add_argument("--P_max", type=float, default=1, help="最大光功率 (W)")
    parser.add_argument("--adc_bits", type=int, default=8, help="ADC 量化位数")
    parser.add_argument("--outdir", default="output_video_model44", help="输出目录")
    parser.add_argument("--pixel_var_resp", type=float, default=PhotodetectorArray.DEFAULT_VARIATION["responsivity_cv"],
                        help="像素 responsivity 相对标准差")
    parser.add_argument("--pixel_var_eta", type=float, default=PhotodetectorArray.DEFAULT_VARIATION["eta_sigma"],
                        help="像素 eta 绝对标准差")
    parser.add_argument("--pixel_var_tau", type=float, default=PhotodetectorArray.DEFAULT_VARIATION["tau_cv"],
                        help="像素时间常数相对标准差")
    parser.add_argument("--pixel_var_dark", type=float, default=PhotodetectorArray.DEFAULT_VARIATION["dark_current_cv"],
                        help="像素暗电流相对标准差")
    parser.add_argument("--pixel_var_noise", type=float, default=PhotodetectorArray.DEFAULT_VARIATION["thermal_noise_cv"],
                        help="像素热噪声底相对标准差")
    parser.add_argument("--max_output_frames", type=int, default=10,
                        help="最多生成的输出帧数")
    parser.add_argument("--disable_noise", action="store_true",
                        help="关闭 shot noise 和 thermal noise")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    variation_cfg = {
        "responsivity_cv": args.pixel_var_resp,
        "eta_sigma": args.pixel_var_eta,
        "tau_cv": args.pixel_var_tau,
        "dark_current_cv": args.pixel_var_dark,
        "thermal_noise_cv": args.pixel_var_noise,
    }
    noise_cfg = None
    if args.disable_noise:
        noise_cfg = {
            "i_thermal": 0.0,
            "bandwidth": PhotodetectorArray.DEFAULT_NOISE["bandwidth"],
            "shot_noise": False,
        }

    # --- Step 1: Load video ---
    print("=" * 60)
    print("Step 1: 读取视频")
    frames_P, fps_orig = load_video_as_power(
        args.video, target_size=(args.size, args.size), P_max=args.P_max,
        max_output_frames=args.max_output_frames, fps_output=args.fps_output
    )

    # --- Step 2: Temporal interpolation ---
    print("\n" + "=" * 60)
    print(f"Step 2: 时间插值 ({fps_orig:.1f} Hz → {args.fps_sim:.0f} Hz)")
    frames_interp, t_sim = interpolate_frames(frames_P, fps_orig, args.fps_sim)
    dt_sim = t_sim[1] - t_sim[0]
    print(f"  {len(frames_interp)} 仿真步, dt={dt_sim*1e3:.3f} ms, "
          f"T={t_sim[-1]:.2f} s")
    if args.max_output_frames is not None:
        print(f"  最多输出帧数 = {args.max_output_frames}")
    print("  像素间 variation:")
    print(f"    responsivity_cv = {variation_cfg['responsivity_cv']:.3f}")
    print(f"    eta_sigma       = {variation_cfg['eta_sigma']:.3f}")
    print(f"    tau_cv          = {variation_cfg['tau_cv']:.3f}")
    print(f"    dark_current_cv = {variation_cfg['dark_current_cv']:.3f}")
    print(f"    thermal_noise_cv= {variation_cfg['thermal_noise_cv']:.3f}")

    array_tia = PhotodetectorArray(
        args.size, args.size, variation_params=variation_cfg, noise_params=noise_cfg
    )
    print_nonideal_parameter_summary(array_tia, args, fps_orig, dt_sim)

    # --- Step 3: Ideal response (no non-idealities, used as the reference) ---
    print("\n" + "=" * 60)
    print("Step 3: 生成理想响应")
    R_total = (PhotodetectorArray.DEFAULT_PARAMS["R_fast"]
               + PhotodetectorArray.DEFAULT_PARAMS["R_slow"])
    dt_output = 1.0 / args.fps_output
    ideal_frames = []
    t_now = 0
    next_read = dt_output
    accum = np.zeros((args.size, args.size))
    for P in frames_interp:
        accum += R_total * P * dt_sim
        t_now += dt_sim
        if t_now >= next_read - dt_sim * 0.5:
            ideal_frames.append(accum.copy())
            accum[:] = 0
            next_read += dt_output
            if args.max_output_frames is not None and len(ideal_frames) >= args.max_output_frames:
                break
    print(f"  {len(ideal_frames)} 帧")

    # --- Step 4: TIA-mode simulation ---
    print("\n" + "=" * 60)
    print("Step 4: TIA 模式仿真")
    readout_tia = ReadoutTIA(array_tia)
    tia_frames = simulate_video(frames_interp, t_sim, array_tia,
                                readout_tia, args.fps_output,
                                max_output_frames=args.max_output_frames)

    # --- Step 5: Integration-mode simulation ---
    print("\n" + "=" * 60)
    print("Step 5: 积分模式仿真")
    array_integ = PhotodetectorArray(
        args.size, args.size, variation_params=variation_cfg, noise_params=noise_cfg
    )
    readout_integ = ReadoutIntegration(array_integ)
    integ_frames = simulate_video(frames_interp, t_sim, array_integ,
                                  readout_integ, args.fps_output,
                                  max_output_frames=args.max_output_frames)

    # --- Step 6: ADC-mode simulation ---
    print("\n" + "=" * 60)
    print(f"Step 6: ADC {args.adc_bits}-bit 模式仿真")
    array_adc = PhotodetectorArray(
        args.size, args.size, variation_params=variation_cfg, noise_params=noise_cfg
    )
    readout_adc = ReadoutADC(array_adc, n_bits=args.adc_bits)
    adc_frames = simulate_video(frames_interp, t_sim, array_adc,
                                readout_adc, args.fps_output,
                                max_output_frames=args.max_output_frames)

    # --- Step 7: Write output videos ---
    print("\n" + "=" * 60)
    print("Step 7: 生成输出视频")
    fps_out = args.fps_output

    frames_to_video(ideal_frames, outdir / "ideal.mp4", fps_out)
    frames_to_video(tia_frames, outdir / "tia.mp4", fps_out)
    frames_to_video(integ_frames, outdir / "integration.mp4", fps_out)
    frames_to_video(adc_frames, outdir / f"adc_{args.adc_bits}bit.mp4", fps_out)

    # 2x2 comparison
    make_comparison_video(ideal_frames, tia_frames, integ_frames, adc_frames,
                          outdir / "comparison.mp4", fps_out)

    print("\n" + "=" * 60)
    print("完成! 输出文件:")
    for f in sorted(outdir.glob("*.mp4")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
