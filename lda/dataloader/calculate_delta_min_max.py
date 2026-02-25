# 
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
import argparse
from lda.utils.rotation_convert import calculate_delta_eef

# modality.json
modality_data ={
    "state": {
        "left_eef_position": {
            "original_key": "eef.left.wrist",
            "start": 0,
            "end": 3, 
            "absolute": True
        },
        "left_eef_rotation": {
            "original_key": "eef.left.wrist",
            "start": 3,
            "end": 6,
            "absolute": True,
            "rotation_type": "euler_angles_rpy"
        },
        "right_eef_position": {
            "original_key": "eef.right.wrist",
            "start": 0,
            "end": 3,
            "absolute": True
        },
        "right_eef_rotation": {
            "original_key": "eef.right.wrist",
            "start": 3,
            "end": 6,
            "absolute": True,
            "rotation_type": "euler_angles_rpy"
        },
        "left_gripper": {
            "original_key": "eef.left.hand",
            "start": 0,
            "end": 1,
            "absolute": True
        },
        "right_gripper": {
            "original_key": "eef.right.hand",
            "start": 0,
            "end": 1,
            "absolute": True
        }
    },
    "action": {
        "left_eef_position": {
            "original_key": "eef.left.wrist",
            "start": 0,
            "end": 3,
            "absolute": True
        },
        "left_eef_rotation": {
            "original_key": "eef.left.wrist",
            "start": 3,
            "end": 6,
            "absolute": True,
            "rotation_type": "euler_angles_rpy"
        },
        "right_eef_position": {
            "original_key": "eef.right.wrist",
            "start": 0,
            "end": 3,
            "absolute": True
        },
        "right_eef_rotation": {
            "original_key": "eef.right.wrist",
            "start": 3,
            "end": 6,
            "absolute": True,
            "rotation_type": "euler_angles_rpy"
        },
        "left_gripper": {
            "original_key": "eef.left.hand",
            "start": 0,
            "end": 1,
            "absolute": True
        },
        "right_gripper": {
            "original_key": "eef.right.hand",
            "start": 0,
            "end": 1,
            "absolute": True
        }
    },
    "video": {
        "top_head": {
            "original_key": "observation.images.top_head"
        }
    },
    "annotation": {
        "language.action_text": {
            "original_key": "task_index"
        }
    }
}

def process_parquet_file(args):
    pq_path, window_size, samples_per_file, worker_id, collect_full_stats = args

    try:
        df = pd.read_parquet(pq_path)
        left_delta_result = {}
        right_delta_result = {}
        # ===== 1. delta_eef =====
        delta_result = None
        if "eef.left.wrist" in df:
            left = np.stack(df["eef.left.wrist"].to_numpy())   # (T, 6)
            length = len(left)
            if length >= window_size:
                rng = np.random.default_rng(seed=1234 + worker_id)
                num_samples = min(samples_per_file, length - window_size + 1)
                starts = rng.choice(length - window_size + 1, size=num_samples, replace=False)

                local = {
                    "min_left": np.full(6, np.inf, dtype=np.float64),
                    "max_left": np.full(6, -np.inf, dtype=np.float64),
                    "sum_left": np.zeros(6, dtype=np.float64),
                    "sumsq_left": np.zeros(6, dtype=np.float64),
                    "count_left": 0,
                    "samples_left": [],
                }

                for start in starts:
                    l_slice = left[start:start + window_size]

                    d_left = calculate_delta_eef(l_slice)

                    if len(d_left) > 0:
                        local["min_left"] = np.minimum(local["min_left"], d_left.min(axis=0))
                        local["max_left"] = np.maximum(local["max_left"], d_left.max(axis=0))
                        local["sum_left"] += d_left.sum(axis=0)
                        local["sumsq_left"] += (d_left ** 2).sum(axis=0)
                        local["count_left"] += len(d_left)
                        local["samples_left"].append(d_left)

                left_delta_result = local
        if "eef.right.wrist" in df:
            right = np.stack(df["eef.right.wrist"].to_numpy()) # (T, 6)
            length = len(right)
            if length >= window_size:
                rng = np.random.default_rng(seed=1234 + worker_id)
                num_samples = min(samples_per_file, length - window_size + 1)
                starts = rng.choice(length - window_size + 1, size=num_samples, replace=False)

                local = {
                    "min_right": np.full(6, np.inf, dtype=np.float64),
                    "max_right": np.full(6, -np.inf, dtype=np.float64),
                    "sum_right": np.zeros(6, dtype=np.float64),
                    "sumsq_right": np.zeros(6, dtype=np.float64),
                    "count_right": 0,
                    "samples_right": [],
                }

                for start in starts:
                    r_slice = right[start:start + window_size]

                    d_right = calculate_delta_eef(r_slice)

                    if len(d_right) > 0:
                        local["min_right"] = np.minimum(local["min_right"], d_right.min(axis=0))
                        local["max_right"] = np.maximum(local["max_right"], d_right.max(axis=0))
                        local["sum_right"] += d_right.sum(axis=0)
                        local["sumsq_right"] += (d_right ** 2).sum(axis=0)
                        local["count_right"] += len(d_right)
                        local["samples_right"].append(d_right)

                right_delta_result = local
        delta_result = left_delta_result | right_delta_result
        # ===== 2. collect all data statistics =====
        full_columns = {}
        if collect_full_stats:
            for col in df.columns:
                if col.startswith("annotation."):
                    continue
                if col in ['eef.left.wrist', 'eef.right.wrist']:
                    continue

                try:
                    series = df[col]
                    if series.dtype == 'object':
                        # try to convert to float32 array
                        arr_list = []
                        for x in series:
                            if x is None:
                                continue
                            try:
                                arr = np.asarray(x, dtype=np.float32)
                                if np.any(np.isnan(arr)):
                                    continue  # skip nan-containing rows
                                arr_list.append(arr)
                            except Exception:
                                continue
                        if not arr_list:
                            continue
                        # concatenate to (N, D) or (N,)
                        values = np.stack(arr_list) if arr_list[0].ndim > 0 else np.array(arr_list)
                    else:
                        # numerical columns
                        values = series.dropna().to_numpy(dtype=np.float32)
                        if values.size == 0:
                            continue

                    full_columns[col] = values
                except Exception as e:
                    # silently skip problematic columns
                    continue

        return {
            "delta": delta_result,
            "full_columns": full_columns
        }

    except Exception as e:
        print(f"[Worker {worker_id}] Error processing {pq_path}: {e}")
        return None


def process_single_task(parquet_paths, save_path, window_size, samples_per_file, collect_full_stats, num_workers):
    tasks = [
            (pq, window_size, samples_per_file, i, collect_full_stats)
            for i, pq in enumerate(parquet_paths)
        ]

    print(f"Processing {len(tasks)} parquet files with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_parquet_file, tasks),
            total=len(tasks),
            desc="Processing files"
        ))

    # ===== aggregate delta stats =====
    sides = ["left", "right"]
    global_delta = {
        "min": {s: np.zeros(6) for s in sides},
        "max": {s: np.zeros(6) for s in sides},
        "sum": {s: np.zeros(6) for s in sides},
        "sumsq": {s: np.zeros(6) for s in sides},
        "count": {s: 0 for s in sides},
        "samples": {s: [] for s in sides},
    }

    # ===== aggregate full columns =====
    full_column_samples = {}  # col -> list of arrays

    for res in results:
        if res is None:
            continue

        # aggregate delta
        delta_res = res["delta"]
        if delta_res is not None:
            for side in sides:
                if f"min_{side}" in delta_res:
                    global_delta["min"][side] = np.minimum(global_delta["min"][side], delta_res[f"min_{side}"])
                    global_delta["max"][side] = np.maximum(global_delta["max"][side], delta_res[f"max_{side}"])
                    global_delta["sum"][side] += delta_res[f"sum_{side}"]
                    global_delta["sumsq"][side] += delta_res[f"sumsq_{side}"]
                    global_delta["count"][side] += delta_res[f"count_{side}"]
                    global_delta["samples"][side].extend(delta_res[f"samples_{side}"])

        # aggregate full columns
        if collect_full_stats:
            for col, values in res["full_columns"].items():
                if col not in full_column_samples:
                    full_column_samples[col] = []
                full_column_samples[col].append(values)

    # ===== Finalize delta =====
    def finalize_delta(side: str):
        cnt = global_delta["count"][side]
        if cnt == 0:
            return {k: np.zeros(6).tolist() for k in ["min", "max", "mean", "std", "q01", "q99"]}
        mean = global_delta["sum"][side] / cnt
        var = np.maximum(global_delta["sumsq"][side] / cnt - mean**2, 0.0)
        std = np.sqrt(var)
        min_val = global_delta["min"][side]
        max_val = global_delta["max"][side]

        stacked = np.concatenate(global_delta["samples"][side], axis=0) if global_delta["samples"][side] else np.zeros((0, 6))
        q01 = np.quantile(stacked, 0.01, axis=0).tolist() if len(stacked) > 0 else [None] * 6
        q99 = np.quantile(stacked, 0.99, axis=0).tolist() if len(stacked) > 0 else [None] * 6

        return {
            "min": min_val.tolist(),
            "max": max_val.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "q01": q01,
            "q99": q99,
        }

    stats = {
        "eef.left.wrist": finalize_delta("left"),
        "eef.right.wrist": finalize_delta("right"),
    }

    # ===== Finalize full dataset statistics =====
    if collect_full_stats:
        full_stats = {}
        for col, chunks in full_column_samples.items():
            try:
                all_vals = np.concatenate(chunks, axis=0)
                if all_vals.size == 0:
                    continue

                # calculate statistics for each dimension (support 1D or 2D)
                axis = 0 if all_vals.ndim > 1 else None
                full_stats[col] = {
                    "mean": np.atleast_1d(np.mean(all_vals, axis=axis)).tolist(),
                    "std": np.atleast_1d(np.std(all_vals, axis=axis)).tolist(),
                    "min": np.atleast_1d(np.min(all_vals, axis=axis)).tolist(),
                    "max": np.atleast_1d(np.max(all_vals, axis=axis)).tolist(),
                    "q01": np.atleast_1d(np.quantile(all_vals, 0.01, axis=axis)).tolist(),
                    "q99": np.atleast_1d(np.quantile(all_vals, 0.99, axis=axis)).tolist(),
                }
            except Exception as e:
                print(f"Error finalizing full stats for {col}: {e}")
                continue
        for key, value in full_stats.items():
            stats[key] = value

    # Save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"✅ Saved combined stats to {save_path}")
    
    # Save modality.json
    modality_path = save_path.parent / "modality_eef.json"
    with open(modality_path, "w") as f:
        json.dump(modality_data, f, indent=4)
    print(f"✅ Saved modality config to {modality_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="stats_eef.json")
    parser.add_argument("--window_size", type=int, default=17)
    parser.add_argument("--samples_per_file", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--collect_full_stats", type=str, default=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)

    # get all data directories (recursively)
    data_dirs = [p for p in dataset_root.rglob("data") if p.is_dir()]

    if not data_dirs:
        raise ValueError(f"No 'data' directories found under {dataset_root}")

    # group by task_root (task_root is the parent directory of data)
    from collections import defaultdict
    task_to_parquet_paths = defaultdict(list)

    for data_dir in data_dirs:
        task_root = data_dir.parent
        parquet_files = list(data_dir.rglob("*.parquet"))
        if parquet_files:
            task_to_parquet_paths[task_root].extend(parquet_files)

    # if only one task is found (and its parent is dataset_root), it can be regarded as a single task mode
    if len(task_to_parquet_paths) == 1:
        task_root, parquet_paths = next(iter(task_to_parquet_paths.items()))
        if task_root == dataset_root:
            # it is indeed a flat structure: dataset_root/data/...
            save_path = Path(dataset_root) / "meta" / args.save_path
            process_single_task(sorted(parquet_paths), save_path, args.window_size, args.samples_per_file, args.collect_full_stats, args.num_workers)
        else:
            # dataset_root/taskX/data/...
            task_name = task_root.name
            save_path = task_root / "meta" / args.save_path
            process_single_task(sorted(parquet_paths), save_path, args.window_size, args.samples_per_file, args.collect_full_stats, args.num_workers)
    else:
        for task_root, parquet_paths in task_to_parquet_paths.items():
            try:
                task_name = task_root.name
                save_path = task_root / "meta" / args.save_path
                process_single_task(
                    sorted(parquet_paths),
                    save_path,
                    args.window_size,
                    args.samples_per_file,
                    args.collect_full_stats,
                    args.num_workers
                )
            except Exception as e:
                print(f"Error processing task {task_name} (root={task_root}): {e}")
                continue
if __name__ == "__main__":
    main()