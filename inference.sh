#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="configs/self_forcing_dmd.yaml"
MODEL_PATH="checkpoints/ema_model.pt"
WAN_MODEL_PATH="..models/Wan2.1-T2V-1.3B"
PROMPTS_PATH="eval_caption_multishot_t2v_100_infinity_rope_prompts.txt"
OUTPUT_ROOT="videos/eval_caption_multishot_t2v_100"
SEED=0
NUM_SAMPLES=1
USE_EMA=1
SHOTS_PER_VIDEO=6
SHOT_SECONDS=5
MODEL_FPS=16
TEMPORAL_COMPRESSION=4
NUM_FRAME_PER_BLOCK=""
DO_UPLOAD=1
HF_REPO_ID="Orannue/Baseline_results"
HF_REPO_TYPE="dataset"
HF_UPLOAD_PATH="eval_caption_multishot_t2v_100/infinity_rope"
HF_TOKEN="${HF_TOKEN:-}"

usage() {
    cat <<'EOF'
Usage:
  bash inference.sh --model_path /path/to/checkpoint.pt [options]

Options:
  --model_path, --checkpoint_path PATH   Model checkpoint path. Default: checkpoints/ema_model.pt
  --wan_model_path PATH                  Wan2.1-T2V-1.3B base model folder. Default: wan_models/Wan2.1-T2V-1.3B
  --config_path PATH                     Config path. Default: configs/self_forcing_dmd.yaml
  --prompts_path PATH                    Prompt txt file. Default: eval_caption_multishot_t2v_100_infinity_rope_prompts.txt
  --output_root PATH                     Final output root. Default: videos/eval_caption_multishot_t2v_100
  --seed INT                             Seed. Default: 0
  --num_samples INT                      Samples per prompt. Default: 1
  --shots_per_video INT                  Number of shots to split from each full video. Default: 6
  --shot_seconds FLOAT                   Seconds per split shot. Default: 5
  --model_fps FLOAT                      FPS used by inference.py duration logic. Default: 16
  --temporal_compression INT             VAE temporal compression. Default: 4
  --num_frame_per_block INT              Override config num_frame_per_block. Default: read from config
  --no_ema                               Load checkpoint key "generator" instead of "generator_ema"
  --upload / --no_upload                 Upload final output folder to HuggingFace. Default: upload
  --hf_repo_id ID                        HuggingFace repo id. Default: Orannue/multishot_long_video
  --hf_repo_type TYPE                    HuggingFace repo type. Default: model
  --hf_upload_path PATH                  HuggingFace destination path. Default: infinity_rope/eval_caption_multishot_t2v_100
  --hf_path_in_repo PATH                 Alias for --hf_upload_path
  --hf_token TOKEN                       HuggingFace token. Default: HF_TOKEN env or cached login
  -h, --help                             Show this help message

Final layout:
  ${output_root}/video1/full.mp4
  ${output_root}/video1/shot1.mp4
  ${output_root}/video1/shot2.mp4
  ...
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path|--checkpoint_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --wan_model_path|--wan_path)
            WAN_MODEL_PATH="$2"
            shift 2
            ;;
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --prompts_path|--data_path)
            PROMPTS_PATH="$2"
            shift 2
            ;;
        --output_root|--output_folder)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --shots_per_video)
            SHOTS_PER_VIDEO="$2"
            shift 2
            ;;
        --shot_seconds)
            SHOT_SECONDS="$2"
            shift 2
            ;;
        --model_fps)
            MODEL_FPS="$2"
            shift 2
            ;;
        --temporal_compression)
            TEMPORAL_COMPRESSION="$2"
            shift 2
            ;;
        --num_frame_per_block)
            NUM_FRAME_PER_BLOCK="$2"
            shift 2
            ;;
        --no_ema)
            USE_EMA=0
            shift
            ;;
        --upload)
            DO_UPLOAD=1
            shift
            ;;
        --no_upload)
            DO_UPLOAD=0
            shift
            ;;
        --hf_repo_id)
            HF_REPO_ID="$2"
            shift 2
            ;;
        --hf_repo_type)
            HF_REPO_TYPE="$2"
            shift 2
            ;;
        --hf_upload_path|--hf_path_in_repo)
            HF_UPLOAD_PATH="$2"
            shift 2
            ;;
        --hf_token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ "$NUM_SAMPLES" != "1" ]]; then
    echo "This layout script currently expects --num_samples 1, got ${NUM_SAMPLES}." >&2
    exit 1
fi

EMA_ARGS=()
MODEL_TAG="regular"
if [[ "$USE_EMA" == "1" ]]; then
    EMA_ARGS=(--use_ema)
    MODEL_TAG="ema"
fi

RAW_OUTPUT="${OUTPUT_ROOT}/_raw_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RAW_OUTPUT"

python inference.py \
    --config_path "$CONFIG_PATH" \
    --checkpoint_path "$MODEL_PATH" \
    --wan_model_path "$WAN_MODEL_PATH" \
    --output_folder "$RAW_OUTPUT" \
    --data_path "$PROMPTS_PATH" \
    --seed "$SEED" \
    --num_samples "$NUM_SAMPLES" \
    --save_with_index \
    "${EMA_ARGS[@]}"

python - "$RAW_OUTPUT" "$OUTPUT_ROOT" "$PROMPTS_PATH" "$CONFIG_PATH" "$SEED" "$MODEL_TAG" "$SHOTS_PER_VIDEO" "$SHOT_SECONDS" "$MODEL_FPS" "$TEMPORAL_COMPRESSION" "$NUM_FRAME_PER_BLOCK" <<'PY'
import math
import re
import sys
from pathlib import Path

from torchvision.io import read_video, write_video

raw_output = Path(sys.argv[1])
output_root = Path(sys.argv[2])
prompts_path = Path(sys.argv[3])
config_path = Path(sys.argv[4])
seed = sys.argv[5]
model_tag = sys.argv[6]
shots_per_video = int(sys.argv[7])
shot_seconds = float(sys.argv[8])
model_fps = float(sys.argv[9])
temporal_compression = int(sys.argv[10])
num_frame_per_block_arg = sys.argv[11]

def read_num_frame_per_block(path):
    if num_frame_per_block_arg:
        return int(num_frame_per_block_arg)
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*num_frame_per_block\s*:\s*(\d+)\s*$", text)
    if not match:
        return 1
    return int(match.group(1))

num_frame_per_block = read_num_frame_per_block(config_path)
frames_per_generation_block = temporal_compression * num_frame_per_block

def parse_durations(prompt):
    prompt_part = prompt.split(";")[0]
    scene_parts = [part.strip() for part in prompt_part.split("|")]
    durations = []
    for scene_part in scene_parts:
        match = re.search(r"\[(\d+\.?\d*)\s*s#?\]", scene_part)
        if not match:
            return None
        durations.append(float(match.group(1)))
    return durations

def calculate_total_blocks(total_duration):
    total_output_frames = int(total_duration * model_fps)
    base_latent_frames = total_output_frames // temporal_compression
    latent_frames = math.ceil(base_latent_frames / num_frame_per_block) * num_frame_per_block
    latent_frames = max(latent_frames, num_frame_per_block)
    return latent_frames // num_frame_per_block

def fit_scene_blocks_to_generation_length(block_counts, total_blocks):
    if len(block_counts) == 1:
        return [total_blocks]
    requested_total = sum(block_counts)
    if requested_total == total_blocks:
        return block_counts

    boundaries = []
    cumulative = 0
    for count in block_counts[:-1]:
        cumulative += count
        boundaries.append(round(cumulative * total_blocks / requested_total))

    fitted_counts = []
    previous = 0
    num_scenes = len(block_counts)
    for i, boundary in enumerate(boundaries):
        remaining_scenes = num_scenes - i - 1
        min_boundary = previous + 1
        max_boundary = total_blocks - remaining_scenes
        boundary = min(max(boundary, min_boundary), max_boundary)
        fitted_counts.append(boundary - previous)
        previous = boundary

    fitted_counts.append(total_blocks - previous)
    return fitted_counts

def scene_boundaries_for_prompt(prompt):
    durations = parse_durations(prompt)
    if durations is None:
        shot_frames = round(shot_seconds * model_fps)
        return [i * shot_frames for i in range(shots_per_video + 1)], [shot_frames] * shots_per_video

    block_counts = [
        max(1, int((duration * model_fps) / frames_per_generation_block))
        for duration in durations
    ]
    total_blocks = calculate_total_blocks(sum(durations))
    block_counts = fit_scene_blocks_to_generation_length(block_counts, total_blocks)

    boundaries = [0]
    cumulative_blocks = 0
    for block_count in block_counts:
        cumulative_blocks += block_count
        boundaries.append(cumulative_blocks * frames_per_generation_block)
    return boundaries, [count * frames_per_generation_block for count in block_counts]

with prompts_path.open("r", encoding="utf-8") as f:
    prompts = [line.rstrip("\n") for line in f]

for line_idx, _ in enumerate(prompts):
    json_idx = line_idx + 1
    src = raw_output / f"{line_idx}-{seed}_{model_tag}.mp4"
    if not src.exists():
        matches = sorted(raw_output.glob(f"{line_idx}-*_{model_tag}.mp4"))
        if not matches:
            raise FileNotFoundError(f"No generated video found for prompt line {line_idx}: {src}")
        src = matches[0]

    out_dir = output_root / f"video{json_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    video, _, info = read_video(str(src), pts_unit="sec", output_format="THWC")
    if video.numel() == 0:
        raise RuntimeError(f"Generated video is empty: {src}")

    fps = info.get("video_fps", 16)
    write_video(str(out_dir / "full.mp4"), video, fps=fps)

    total_frames = video.shape[0]
    boundaries, shot_frame_counts = scene_boundaries_for_prompt(prompts[line_idx])
    if boundaries[-1] > total_frames:
        raise RuntimeError(
            f"Generated video{json_idx} has {total_frames} frames, "
            f"but prompt-aligned scene boundaries require {boundaries[-1]} frames. "
            f"num_frame_per_block={num_frame_per_block}, "
            f"frames_per_generation_block={frames_per_generation_block}."
        )
    for shot_idx in range(shots_per_video):
        start = boundaries[shot_idx]
        end = boundaries[shot_idx + 1]
        if end <= start:
            raise RuntimeError(f"Invalid shot boundary for video{json_idx}: {start}..{end}")
        write_video(str(out_dir / f"shot{shot_idx + 1}.mp4"), video[start:end], fps=fps)

    print(
        f"video{json_idx}: {out_dir / 'full.mp4'} + {shots_per_video} shots "
        f"(prompt-aligned frames per shot: {shot_frame_counts})"
    )
PY

if [[ "$DO_UPLOAD" == "1" ]]; then
    python - "$OUTPUT_ROOT" "$HF_REPO_ID" "$HF_REPO_TYPE" "$HF_UPLOAD_PATH" "$HF_TOKEN" <<'PY'
import os
import sys
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi

output_root = Path(sys.argv[1])
repo_id = sys.argv[2]
repo_type = sys.argv[3]
path_in_repo = sys.argv[4].replace("\\", "/")
token = sys.argv[5] or os.environ.get("HF_TOKEN") or None

print(f"Starting upload to HuggingFace Hub: {output_root} -> {repo_id}/{path_in_repo}")
api = HfApi()
api.upload_folder(
    folder_path=str(output_root),
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type=repo_type,
    token=token,
)
print(f"Upload to HuggingFace Hub completed: {output_root}")
PY
fi
