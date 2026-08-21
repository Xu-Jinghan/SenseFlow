import ast
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.utils.data
import torch.nn as nn
from datetime import datetime
from subprocess import call

PROJECT_ROOT = Path(__file__).resolve().parent
for extra_path in (PROJECT_ROOT, PROJECT_ROOT / "pytorch-quantization"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from pytorch_quantization.utils import misc, make_path, hook
import generate_sensor_verification_images_video_sequence as sensor_visualization_pipeline
from quantize import quantize_model, evaluate
from dataset import get_imagenet, get_cifar10, get_cifar100, get_sensor_loader
LOCAL_DATA_PATH = PROJECT_ROOT.parent.parent / '.datasets'
DEFAULT_CIM_NONIDEAL_PRESET = {
    "conductance_variation": 0.1, #0.05
    "read_noise": 0.05,#0.01
    "t": 10.0,#0.05
    "v": 0.02,
    "rate_stuck_0": 1e-5,
    "rate_stuck_1": 1e-5,
}
VIDEO_SEQUENCE_SENSOR_DEFAULTS = {
    "sensor_power_max": 0.1,
    "sensor_post_norm": "auto",
    "sensor_shot_noise": 0,
    "sensor_i_thermal": 0.0,
    "sensor_video_fps": 50.0,
}


def _default_data_path():
    if LOCAL_DATA_PATH.exists():
        return str(LOCAL_DATA_PATH)
    return '/usr/scratch1/datasets/'


def _cli_option_provided(argv, name):
    flag = f"--{name}"
    return any(token == flag or token.startswith(flag + "=") for token in argv)


def _apply_sensor_backend_defaults(args, argv):
    changes = []
    if args.dataset != 'sensor' or args.sensor_backend != 'video_sequence':
        return changes

    for name, value in VIDEO_SEQUENCE_SENSOR_DEFAULTS.items():
        if _cli_option_provided(argv, name):
            continue
        if getattr(args, name) != value:
            setattr(args, name, value)
            changes.append((name, value))
    return changes

def parse_args():
    parser = argparse.ArgumentParser()
    # vgg8 (cifar10) software baseline: 89.66%
    # resnet18 (cifar100) software baseline: 75.59%
    # resnet50 (imagenet) software baseline (quantized): 80.012% (79.972%)
    parser.add_argument('--dataset', default='cifar10', help='cifar10|cifar100|imagenet|sensor')
    parser.add_argument('--model', default='vgg8', help='vgg8|DenseNet40|resnet18|resnet50|swin_t')
    parser.add_argument('--data_path', default=_default_data_path(), help='path to saved datasets')
    parser.add_argument('--model_path', default='./models/', help='path to saved models')
    parser.add_argument('--test_name', default='test', help='test name')

    parser.add_argument('--mode', default='TensorRT', help='Quantization mode, only TensorRT supported for now')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use. Only single GPU support for now')
    parser.add_argument('--batch_size', type=int, default=40, help='input batch size for inference (default: 64)')
    parser.add_argument('--calib_batch_size', type=int, default=128, help='input batch size for calibrating quantization (default: 128)')
    parser.add_argument('--num_workers', type=int, default=None, help='dataloader workers for non-sensor CIFAR datasets; None keeps dataset defaults')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1)')
    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')

    # Hardware parameters
    # If you do not want to consider hardware effects, set hardware=0
    parser.add_argument('--hardware',type=int, default=1, help='run hardware inference simulation')
    parser.add_argument('--ppa', type=int, default=1, help='run power, performance, and area analysis (C++)')
    parser.add_argument('--num_batches', type=int, default=-1, help='number of batches to run in hardware inference simulation, -1 for full dataset')
    parser.add_argument('--sub_array', type=str, default="[128, 128]", help='size of subArray (e.g. 128x128)')
    parser.add_argument('--parallel_read', type=int, default=128, help='number of rows read in parallel (<= subArray e.g. 32)')
    parser.add_argument('--weight_precision', type=int, default=8, help='number of bits to quantize the weights to (integer)')
    parser.add_argument('--input_precision', type=int, default=8, help='number of bits to quantize the inputs to (integer)')
    parser.add_argument('--dac_precision', type=int, default=1, help='DAC precision (e.g. 1-bit)')
    parser.add_argument('--adc_precision', type=int, default=7, help='ADC precision (e.g. 7-bit)')
    parser.add_argument('--bitcell', type=int, default=1, help='cell precision (e.g. 4-bit/cell)')
    parser.add_argument('--mem_type', type=str, default='resistive', help='memory cell type: resisitive | capacitive')
    parser.add_argument('--off_state', type=float, default=6e-3, help='device off state (conductance (S) or capacitance (F))')
    parser.add_argument('--on_state', type=float, default=6e-13*17, help='device on conductance')
    parser.add_argument('--mem_states_file', type=str, default="mem_states.csv", help='path to .csv file containing mean and std for each memory state')
    parser.add_argument('--conductance_variation', type=float, default=0.0, help='relative std applied to each conductance state for device-to-device variation when CSV std is unavailable')
    parser.add_argument('--vdd', type=float, default=1, help='supply voltage')
    parser.add_argument('--read_noise', type=float, default=0.0, help='std of conductance read noise, derive from SPICE or experiment, or use to test reliability')
    parser.add_argument('--output_noise', type=float, default=0.0, help='std of output voltage noise, derive from SPICE or experiment, or use to test reliability')
    # set output_noise=-1 to load the output std from external csv
    # set mean with relative deviation and std with relative value
    parser.add_argument('--output_noise_file', type=str, default="", help='path and file to saved output std')

    # TODO:
    # if do not run the device retention, set t=0, v=0
    parser.add_argument('--t', type=float, default=1, help='retention time')
    parser.add_argument('--v', type=float, default=0.00, help='drift coefficient')
    parser.add_argument('--v_list', type=str, default="[0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005]", help='drift coefficient')
    parser.add_argument('--detect', type=int, default=0, help='if 1, fixed-direction drift, if 0, random drift')
    parser.add_argument('--target', type=float, default=0.0, help='drift target for fixed-direction drift, range 0-1')

    # stuck at fault parameters
    parser.add_argument('--rate_stuck_0', type=float, default=0.00, help='rate of cells stuck at 0, range 0-1, default 0')
    parser.add_argument('--rate_stuck_1', type=float, default=0.00, help='rate of cells stuck at 1, range 0-1, default 0')
    parser.add_argument('--cim_nonideal', type=int, default=1, help='overall switch for CIM nonideal effects; 0 disables variation, drift, read noise, stuck-at faults, and mem-state CSV variation')
    parser.add_argument('--cim_nonideal_preset', type=str, default='light', choices=['off', 'light'], help='preset for enabling CIM nonidealities in hardware inference')

    # TensorRT quantization parameters
    parser.add_argument("--input_calib_method",  type=str, default='max', help='histogram or max')
    parser.add_argument("--weight_calib_method", type=str, default='max', help='histogram or max')
    parser.add_argument("--adc_calib_method",    type=str, default='max', help='histogram or max')
    parser.add_argument("--input_axis", type=int,  default=None, help='axis to quantize (e.g. 0 for batch, 1 for channel, 2 for height, 3 for width)')
    parser.add_argument("--weight_axis", type=int, default=0, help='axis to quantize (e.g. 0 for batch, 1 for channel, 2 for height, 3 for width)')
    parser.add_argument("--adc_axis",    type=int, default=None, help='axis to quantize (e.g. 0 for batch, 1 for channel, 2 for height, 3 for width)')
    parser.add_argument("--adc_quant_method", type=str, default='scale')
    parser.add_argument("--optimize_adc",type=int, default=0)
    parser.add_argument("--adc_enable",type=int, default=1)

    # Sensor loader parameters
    parser.add_argument('--sensor_source_dataset', default='cifar10', help='base dataset for sensor simulation: cifar10|cifar100|imagenet')
    parser.add_argument('--sensor_readout', default='integration', help='sensor readout mode: tia|integration|adc')
    parser.add_argument('--sensor_array_size', type=int, default=0, help='photodetector array size; 0 uses the model input size')
    parser.add_argument('--sensor_target_size', type=int, default=0, help='network input size after sensor readout; 0 infers from model')
    parser.add_argument('--sensor_output_channels', type=int, default=0, help='number of channels fed to the model; 0 infers from model, 3 keeps RGB channels, 1 converts to grayscale')
    parser.add_argument('--sensor_power_max', type=float, default=0.1, help='max optical power mapped from normalized image intensity')
    parser.add_argument('--sensor_params_csv', type=str, default='', help='optional CSV exported by fit_waveform_from_image.py to override photodetector model parameters')
    parser.add_argument('--sensor_spatial_variation_r_pct', type=float, default=None, help='override spatial variation of R (%%) for the sensor frontend, especially video_sequence backend')
    parser.add_argument('--sensor_exposure_time', type=float, default=1.0/30.0, help='static exposure time per image in seconds')
    parser.add_argument('--sensor_fps_sim', type=float, default=1000.0, help='internal photodetector simulation step rate in Hz')
    parser.add_argument('--sensor_adc_bits', type=int, default=8, help='ADC bit width used by ReadoutADC')
    parser.add_argument('--sensor_adc_full_scale', type=float, default=None, help='fixed full-scale charge for ReadoutADC')
    parser.add_argument('--sensor_range_mode', default='auto', help='range scaling applied before model input: auto|minmax|signed|none')
    parser.add_argument('--sensor_percentile_low', type=float, default=1.0, help='low percentile used by range scaling')
    parser.add_argument('--sensor_percentile_high', type=float, default=99.0, help='high percentile used by range scaling')
    parser.add_argument('--sensor_post_norm', default='none', help='optional post scaling normalization: none|auto|cifar10|cifar100|imagenet')
    parser.add_argument('--sensor_backend', default='video_sequence', choices=['static', 'video_sequence'], help='sensor frontend backend used when dataset=sensor')
    parser.add_argument('--sensor_nonideal', type=int, default=1, help='overall switch for sensor-side nonideal effects; 0 uses ideal sensor readout')
    parser.add_argument('--sensor_num_workers', type=int, default=0, help='number of dataloader workers for sensor mode; 0 recommended')
    parser.add_argument('--sensor_rng_seed', type=int, default=42, help='rng seed for pixel variation and noise maps')
    parser.add_argument('--sensor_shot_noise', type=int, default=1, help='enable shot noise in the photodetector model')
    parser.add_argument('--sensor_bandwidth', type=float, default=5000.0, help='sensor readout bandwidth used by the noise model')
    parser.add_argument('--sensor_i_thermal', type=float, default=5e-8, help='thermal noise current floor')
    parser.add_argument('--sensor_video_fps', type=float, default=0.0, help='frame rate for sensor_backend=video_sequence; 0 uses 1/sensor_exposure_time')
    parser.add_argument('--sensor_use_noise_fn', type=int, default=1, help='enable photodetector_model.NOISE_FN in the video_sequence sensor backend')
    parser.add_argument('--sensor_startup_dark_frames', type=int, default=0, help='number of dark frames to precondition the video_sequence sensor backend')
    parser.add_argument('--sensor_pixel_var_resp', type=float, default=0.08, help='pixel responsivity coefficient of variation')
    parser.add_argument('--sensor_pixel_var_eta', type=float, default=0.01, help='pixel eta standard deviation')
    parser.add_argument('--sensor_pixel_var_tau', type=float, default=0.12, help='pixel time-constant coefficient of variation')
    parser.add_argument('--sensor_pixel_var_dark', type=float, default=0.30, help='pixel dark current coefficient of variation')
    parser.add_argument('--sensor_pixel_var_noise', type=float, default=0.20, help='pixel thermal noise coefficient of variation')
    parser.add_argument(
        '--sensor_save_visuals',
        type=int,
        default=1,
        help='when dataset=sensor and sensor_backend=video_sequence, export comparison images and center-pixel waveform matching the video-sequence verification pipeline',
    )
    parser.add_argument(
        '--sensor_visuals_num_images',
        type=int,
        default=0,
        help='number of sequential validation samples to export for sensor visuals; 0 auto-uses current inference sample count when possible',
    )
    parser.add_argument(
        '--sensor_visuals_start_index',
        type=int,
        default=0,
        help='start dataset index for exported sensor visuals',
    )
    parser.add_argument(
        '--sensor_visuals_output_dir',
        type=str,
        default='',
        help='directory for exported sensor visuals; defaults to <logdir>/sensor_visualization',
    )
    parser.add_argument(
        '--sensor_visuals_tile_size',
        type=int,
        default=256,
        help='tile size used by exported sensor comparison images',
    )
    
    argv = sys.argv[1:]
    args = parser.parse_args()
    args._sensor_backend_default_changes = _apply_sensor_backend_defaults(args, argv)

    # Convert string lists to actual lists
    args.sub_array = ast.literal_eval(args.sub_array)
    args.v_list = ast.literal_eval(args.v_list)
    args.mem_states_file = _resolve_optional_path(args.mem_states_file)
    args.output_noise_file = _resolve_optional_path(args.output_noise_file)
    args._cim_nonideal_disabled_changes = []
    if args.cim_nonideal:
        args._cim_nonideal_changes = _apply_cim_nonideal_preset(args)
    else:
        args._cim_nonideal_disabled_changes = _disable_cim_nonidealities(args)
        args._cim_nonideal_changes = []
    return args


def _resolve_optional_path(path_str):
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    candidate = PROJECT_ROOT / path_str
    if candidate.exists():
        return str(candidate)
    return path_str


def _apply_cim_nonideal_preset(args):
    if args.cim_nonideal_preset == 'off':
        return []

    changes = []

    def maybe_set(name, value, expected_current):
        if getattr(args, name) == expected_current:
            setattr(args, name, value)
            changes.append((name, value))

    maybe_set('conductance_variation', DEFAULT_CIM_NONIDEAL_PRESET['conductance_variation'], 0.0)
    maybe_set('read_noise', DEFAULT_CIM_NONIDEAL_PRESET['read_noise'], 0.0)
    if args.t == 1 and args.v == 0.0:
        args.t = DEFAULT_CIM_NONIDEAL_PRESET['t']
        args.v = DEFAULT_CIM_NONIDEAL_PRESET['v']
        changes.append(('t', args.t))
        changes.append(('v', args.v))
    maybe_set('rate_stuck_0', DEFAULT_CIM_NONIDEAL_PRESET['rate_stuck_0'], 0.0)
    maybe_set('rate_stuck_1', DEFAULT_CIM_NONIDEAL_PRESET['rate_stuck_1'], 0.0)
    return changes


def _disable_cim_nonidealities(args):
    changes = []

    def force(name, value):
        if getattr(args, name) != value:
            setattr(args, name, value)
            changes.append((name, value))

    if args.cim_nonideal_preset != 'off':
        args.cim_nonideal_preset = 'off'
        changes.append(('cim_nonideal_preset', 'off'))

    force('mem_states_file', "")
    force('conductance_variation', 0.0)
    force('read_noise', 0.0)
    force('output_noise', 0.0)
    force('t', 1.0)
    force('v', 0.0)
    force('rate_stuck_0', 0.0)
    force('rate_stuck_1', 0.0)
    return changes


def _infer_sensor_target_size(model, source_dataset):
    if model == 'cimae':
        return 96
    if model == 'swin_t':
        return 256
    if source_dataset in {'cifar10', 'cifar100'}:
        return 32
    return 224


def _infer_sensor_output_channels(model):
    if model == 'cimae':
        return 1
    return 3


def _resolve_sensor_visuals_num_images(args):
    if args.sensor_visuals_num_images > 0:
        return args.sensor_visuals_num_images
    if args.num_batches > 0:
        return max(1, args.num_batches * args.batch_size)
    return min(max(args.batch_size, 1) * 10, 100)


def _resolve_sensor_visuals_output_dir(args):
    if args.sensor_visuals_output_dir:
        return Path(args.sensor_visuals_output_dir).expanduser()
    return Path(args.logdir) / 'sensor_visualization'


def _maybe_export_sensor_visuals(args):
    if not bool(args.sensor_save_visuals):
        return None

    if args.dataset != 'sensor':
        print('Skipping sensor visualization export: dataset must be sensor.')
        return None

    if args.sensor_backend != 'video_sequence':
        print('Skipping sensor visualization export: currently only sensor_backend=video_sequence is supported.')
        return None

    output_dir = _resolve_sensor_visuals_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_size = args.sensor_target_size or _infer_sensor_target_size(args.model, args.sensor_source_dataset)
    output_channels = args.sensor_output_channels or _infer_sensor_output_channels(args.model)
    array_size = args.sensor_array_size or target_size
    video_fps = args.sensor_video_fps if args.sensor_video_fps > 0 else 1.0 / args.sensor_exposure_time
    num_images = _resolve_sensor_visuals_num_images(args)
    results_json = output_dir / 'sensor_visualization_summary.json'

    params_csv = args.sensor_params_csv or str(sensor_visualization_pipeline.base_pipeline.DEFAULT_PARAMS_CSV)

    visualization_args = SimpleNamespace(
        data_root=args.data_path,
        source_dataset=args.sensor_source_dataset,
        split='test',
        generate_images=1,
        run_eval=0,
        eval_cases=list(sensor_visualization_pipeline.EVAL_CASES),
        batch_size=args.batch_size,
        num_workers=0,
        max_eval_batches=0,
        model_path=None,
        num_classes=0,
        results_json=str(results_json),
        seed=args.seed,
        sensor_rng_seed=args.sensor_rng_seed,
        target_size=target_size,
        output_channels=output_channels,
        post_norm=args.sensor_post_norm,
        num_images=num_images,
        start_index=args.sensor_visuals_start_index,
        array_size=array_size,
        tile_size=args.sensor_visuals_tile_size,
        readout=args.sensor_readout,
        power_max=args.sensor_power_max,
        params_csv=params_csv,
        spatial_variation_r_pct=args.sensor_spatial_variation_r_pct,
        video_fps=video_fps,
        fps_sim=args.sensor_fps_sim,
        adc_bits=args.sensor_adc_bits,
        adc_full_scale=args.sensor_adc_full_scale,
        range_mode=args.sensor_range_mode,
        percentile_low=args.sensor_percentile_low,
        percentile_high=args.sensor_percentile_high,
        i_thermal=args.sensor_i_thermal,
        bandwidth=args.sensor_bandwidth,
        shot_noise=args.sensor_shot_noise,
        use_noise_fn=args.sensor_use_noise_fn,
        startup_dark_frames=args.sensor_startup_dark_frames,
        output_dir=str(output_dir),
        analyze_center_pixel=1,
    )

    print(
        f"Exporting sensor visualizations for {num_images} sample(s) to {output_dir}",
        flush=True,
    )
    return sensor_visualization_pipeline.run_sequence_pipeline(visualization_args)


def main():

    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Warning: CUDA is unavailable, falling back to CPU execution.")
    args.device = str(device)
    
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if args.dataset == 'sensor' or args.logdir != 'log/default':
        misc.ensure_dir(args.logdir)
    else:
        args.logdir = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug','data_path','model_path'])
    misc.logger.init(args.logdir, 'test_log_' + current_time)

    args.logger = misc.logger.info

    misc.ensure_dir(args.logdir)
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================\n")

    if args.dataset == 'sensor':
        print(
            f"Sensor frontend: backend={args.sensor_backend} "
            f"nonideal={'enabled' if args.sensor_nonideal else 'disabled'} "
            f"readout={args.sensor_readout}"
        )
        if args._sensor_backend_default_changes:
            print("Aligned sensor video-sequence defaults with generate_sensor_verification_images_video_sequence.py:")
            for key, value in args._sensor_backend_default_changes:
                print(f"  {key}: {value}")
        if args.sensor_backend == 'video_sequence':
            video_fps = args.sensor_video_fps if args.sensor_video_fps > 0 else 1.0 / args.sensor_exposure_time
            print(
                f"Sensor video sequence: fps={video_fps:.4f} "
                f"noise_fn={'enabled' if args.sensor_use_noise_fn else 'disabled'} "
                f"startup_dark_frames={args.sensor_startup_dark_frames} "
                f"spatial_variation_r_pct={args.sensor_spatial_variation_r_pct}"
            )

    if not args.cim_nonideal:
        if args._cim_nonideal_disabled_changes:
            print("CIM nonidealities disabled; forcing these settings to ideal values:")
            for key, value in args._cim_nonideal_disabled_changes:
                print(f"  {key}: {value}")
        print("Active CIM nonidealities: none.\n")
    elif args.cim_nonideal_preset != 'off':
        if args._cim_nonideal_changes:
            print("Enabled CIM nonideal preset with these effective defaults:")
            for key, value in args._cim_nonideal_changes:
                print(f"  {key}: {value}")
        else:
            print("CIM nonideal preset enabled; using user-provided overrides where specified.")
        print(
            "Active CIM nonidealities: conductance variation, conductance read noise, "
            "retention drift, and stuck-at faults.\n"
        )

    print(f"Running in-memory computing functional simulation of {args.model} on {args.dataset} dataset...")
    #torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)

    criterion = nn.CrossEntropyLoss()

    cifar_loader_kwargs = {}
    if args.num_workers is not None:
        cifar_loader_kwargs["num_workers"] = args.num_workers

    if args.dataset == 'imagenet':
        data_loader_quant = get_imagenet(args.calib_batch_size, args.data_path, train=True, val=False, sample=True, model=args.model)
        data_loader_test = get_imagenet(args.batch_size, args.data_path, train=False, val=True, sample=True, model=args.model)
    elif args.dataset == 'cifar100':
        data_loader_quant = get_cifar100(args.calib_batch_size, args.data_path, train=True, val=False, **cifar_loader_kwargs)
        data_loader_test = get_cifar100(args.batch_size, args.data_path, train=False, val=True, **cifar_loader_kwargs)
    elif args.dataset == 'cifar10':
        data_loader_quant = get_cifar10(args.calib_batch_size, args.data_path, train=True, val=False, **cifar_loader_kwargs)
        data_loader_test = get_cifar10(args.batch_size, args.data_path, train=False, val=True, **cifar_loader_kwargs)
    elif args.dataset == 'sensor':
        data_loader_quant = get_sensor_loader(
            args.calib_batch_size,
            args.data_path,
            train=True,
            val=False,
            sample=True,
            model=args.model,
            args=args,
        )
        data_loader_test = get_sensor_loader(
            args.batch_size,
            args.data_path,
            train=False,
            val=True,
            sample=False,
            model=args.model,
            args=args,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.num_batches == -1:
        args.num_batches = len(data_loader_test)

    # Create layer_record directory and NetWork.csv
    hook.make_records(args)
    args.hook = True
    args.write_network = True

    model = quantize_model(args, criterion, data_loader_quant, data_loader_test)
    
    args.logger("Running inference with detailed hardware simulation...")

    model = model.to(device)

    top1, avg_loss = evaluate(model, args, criterion, data_loader_test, num_batches=args.num_batches, print_freq=1)
    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Top-1 accuracy: {top1:.2f}%")

    sensor_visual_results = _maybe_export_sensor_visuals(args)
    if sensor_visual_results is not None:
        center_analysis = sensor_visual_results.get('center_pixel_analysis')
        image_generation = sensor_visual_results.get('image_generation')
        if center_analysis:
            print(f"Sensor center waveform: {center_analysis['plot_path']}")
        if image_generation:
            print(f"Sensor comparison dir: {image_generation['output_dir']}")

    # Uncomment to write outputs to a file
    # with open(f'results/{args.model}/accuracy/{args.test_name}.csv', 'a') as f:
    #    f.write(f'{args.output_noise},{top1},{avg_loss}\n')
    #    f.write(f'{top1},{avg_loss}\n')

    if args.ppa:
        print(" --- Hardware Properties --- ")
        print("subArray size: ")
        print(args.sub_array)
        print("parallel read: ")
        print(args.parallel_read)
        print("ADC precision: ")
        print(args.adc_precision)
        print("cell precision: ")
        print(args.bitcell)
        print("on/off ratio: ")
        print(args.on_state / args.off_state)

        call(["/bin/bash", './layer_record_'+str(args.model)+'/trace_command.sh'])

if __name__ == '__main__':
    main()
