#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
exec zsh

source activate 
conda activate yolo
mkdir ./挂机


nohup python3 挂机训练.py \
  --data /root/yolodata/datasets/VisDrone/VisDrone.yaml \
  --epochs 300 --batch 16 --imgsz 640 --device 0 --workers 8 \
  --models /root/yolodata/ultralytics/cfg/models/1_4/A5_+DSC3k2X.yaml \
  --cache --zip --zip-overwrite \
  > ./挂机/nohup.out 2>&1 &



&：后台运行
nohup：你断开 SSH，进程也不会死
> ./挂机/nohup.out 2>&1：总输出写到一个文件（可选）

查看进程：

bash
复制代码
ps -ef | grep 挂机训练.py
停止进程（把 PID 换成你实际看到的）：

bash
复制代码
kill -9 PID

"""

import argparse
import os
import sys
import time
import shutil
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# 你用的是 ultralytics.YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    print("❌ 无法导入 ultralytics.YOLO，请先确认已安装 ultralytics")
    print("   pip install ultralytics")
    raise


class TeeTextIO:
    """把 stdout/stderr 同时写到多个文件（比如终端 + log 文件）。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sleep_until(start_at_hhmm: str):
    """
    start_at_hhmm: "23:00" 这种格式。
    若当前时间已过，则睡到明天这个点。
    """
    hh, mm = start_at_hhmm.split(":")
    hh = int(hh)
    mm = int(mm)
    now = datetime.now()
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    delta = (target - now).total_seconds()
    print(f"[{now_str()}] ⏳ 将在 {target.strftime('%Y-%m-%d %H:%M:%S')} 开始训练，等待 {int(delta)} 秒...")
    time.sleep(delta)


def make_zip(folder: Path, zip_path: Path, overwrite: bool = True):
    """
    把 folder 打包成 zip_path（例如 ./挂机.zip）。
    注意：zip_path 不能放在 folder 里面，否则会递归把 zip 自己也打进去。
    """
    folder = folder.resolve()
    zip_path = zip_path.resolve()

    if folder in zip_path.parents:
        raise ValueError(f"zip 输出路径 {zip_path} 不能位于要打包的目录 {folder} 内部。")

    if zip_path.exists():
        if overwrite:
            zip_path.unlink()
        else:
            raise FileExistsError(f"{zip_path} 已存在（可用 --zip-overwrite 覆盖）")

    # shutil.make_archive 需要“去掉 .zip 后缀”的 base_name
    base_name = str(zip_path.with_suffix(""))
    shutil.make_archive(base_name, "zip", str(folder))
    print(f"[{now_str()}] ✅ 已打包: {zip_path}")


def train_one(
    model_cfg: Path,
    data_path: Path,
    out_root: Path,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    workers: int,
    cache: bool,
    amp: bool,
    tee_to_terminal: bool,
):
    """
    训练一个模型 cfg（yaml），并把 ultralytics 的输出写入该实验的 log 文件。
    自动断点续训：检测 last.pt 存在则 resume。
    """
    model_cfg = model_cfg.resolve()
    data_path = data_path.resolve()
    out_root = out_root.resolve()

    exp_name = model_cfg.stem  # A0.yaml -> A0
    runs_dir = out_root / "runs" / "train"
    logs_dir = out_root / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ultralytics 的 project/name 组合决定输出目录
    project_dir = runs_dir
    exp_dir = project_dir / exp_name
    last_ckpt_path = exp_dir / "weights" / "last.pt"
    best_ckpt_path = exp_dir / "weights" / "best.pt"

    exp_log = logs_dir / f"{exp_name}.log"

    print(f"\n[{now_str()}] ===============================")
    print(f"[{now_str()}] 🚀 开始实验: {exp_name}")
    print(f"[{now_str()}] 📄 模型 YAML: {model_cfg}")
    print(f"[{now_str()}] 🗂  数据集 YAML: {data_path}")
    print(f"[{now_str()}] 📁 输出目录: {exp_dir}")
    print(f"[{now_str()}] 🧾 日志文件: {exp_log}")
    print(f"[{now_str()}] ===============================\n")

    # 将本次训练的所有 stdout/stderr（含 YOLO 内部打印）写到 exp_log
    exp_log.parent.mkdir(parents=True, exist_ok=True)
    with exp_log.open("a", buffering=1, encoding="utf-8") as lf:
        lf.write(f"\n\n===== [{now_str()}] START {exp_name} =====\n")
        lf.write(f"model_cfg: {model_cfg}\n")
        lf.write(f"data_path: {data_path}\n")
        lf.write(f"out_dir  : {exp_dir}\n")

        orig_out, orig_err = sys.stdout, sys.stderr
        if tee_to_terminal:
            sys.stdout = TeeTextIO(orig_out, lf)
            sys.stderr = TeeTextIO(orig_err, lf)
        else:
            sys.stdout = lf
            sys.stderr = lf

        try:
            # 自动断点续训
            if last_ckpt_path.exists():
                print(f"[{now_str()}] ✅ 检测到断点，续训: {last_ckpt_path}")
                model = YOLO(str(last_ckpt_path))
                resume_training = True
            else:
                print(f"[{now_str()}] 🆕 新训练，从 YAML 构建: {model_cfg}")
                model = YOLO(str(model_cfg))
                resume_training = False

            # 开始训练
            results = model.train(
                data=str(data_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                workers=workers,
                project=str(project_dir),
                name=exp_name,
                resume=resume_training,
                exist_ok=True,   # 允许覆盖同名文件夹（配合 resume 很常用）
                cache=cache,
                amp=amp,
            )

            print(f"\n[{now_str()}] 🎉 实验完成: {exp_name}")
            print(f"[{now_str()}] best.pt: {best_ckpt_path if best_ckpt_path.exists() else '(未找到 best.pt)'}")
            print(f"[{now_str()}] last.pt: {last_ckpt_path if last_ckpt_path.exists() else '(未找到 last.pt)'}")
            lf.write(f"===== [{now_str()}] END {exp_name} (OK) =====\n")

            return True

        except Exception as e:
            print(f"\n[{now_str()}] ❌ 实验失败: {exp_name}")
            print(f"[{now_str()}] 错误: {repr(e)}")
            print(traceback.format_exc())
            lf.write(f"===== [{now_str()}] END {exp_name} (FAILED) =====\n")
            return False

        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err


def main():
    parser = argparse.ArgumentParser(
        description="多 YAML 顺序挂机训练（Ultralytics YOLO），日志落盘，断线可继续，结束自动打包 ./挂机.zip"
    )

    parser.add_argument("--data", required=True, help="数据集 YAML 路径（例如 /root/.../VisDrone.yaml）")
    parser.add_argument("--models", nargs="+", required=True, help="多个模型 YAML（例如 A0.yaml A1.yaml ...）")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0", help="例如 '0' 或 '0,1' 或 'cpu'")
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--out-dir", type=str, default="./挂机", help="所有产物输出到此目录")
    parser.add_argument("--cache", action="store_true", help="启用 cache=True（会占内存）")
    parser.add_argument("--no-amp", action="store_true", help="禁用 amp（默认启用）")

    parser.add_argument("--start-at", type=str, default="", help="可选：到指定时间再开始，例如 23:00")
    parser.add_argument("--tee", action="store_true", help="同时输出到终端 + log（不加则只写 log）")

    parser.add_argument("--continue-on-error", action="store_true", help="某个实验失败后继续下一个")
    parser.add_argument("--zip", action="store_true", help="全部结束后打包 out-dir 为 zip")
    parser.add_argument("--zip-path", type=str, default="", help="zip 输出路径（默认 ./挂机.zip）")
    parser.add_argument("--zip-overwrite", action="store_true", help="覆盖已有 zip（默认不覆盖）")

    args = parser.parse_args()

    data_path = Path(args.data)
    out_root = Path(args.out_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"data yaml 不存在: {data_path}")

    model_paths = [Path(p) for p in args.models]
    for mp in model_paths:
        if not mp.exists():
            raise FileNotFoundError(f"model yaml 不存在: {mp}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)

    # 总控日志（记录每个实验是否成功）
    runner_log = out_root / "logs" / "runner.log"
    with runner_log.open("a", buffering=1, encoding="utf-8") as rf:
        rf.write(f"\n\n===== [{now_str()}] RUNNER START =====\n")
        rf.write(f"data   : {data_path}\n")
        rf.write(f"models : {[str(p) for p in model_paths]}\n")
        rf.write(f"out_dir: {out_root.resolve()}\n")

    print(f"[{now_str()}] 📌 总控日志: {runner_log}")
    print(f"[{now_str()}] 📌 输出根目录: {out_root.resolve()}")

    # 定时开始（可选）
    if args.start_at:
        sleep_until(args.start_at)

    amp = not args.no_amp

    ok_all = True
    for mp in model_paths:
        ok = train_one(
            model_cfg=mp,
            data_path=data_path,
            out_root=out_root,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
            amp=amp,
            tee_to_terminal=args.tee,
        )

        with (out_root / "logs" / "runner.log").open("a", buffering=1, encoding="utf-8") as rf:
            rf.write(f"[{now_str()}] {mp.name} -> {'OK' if ok else 'FAILED'}\n")

        if not ok:
            ok_all = False
            if not args.continue_on_error:
                print(f"[{now_str()}] 由于实验失败且未开启 --continue-on-error，终止后续训练。")
                break

    with (out_root / "logs" / "runner.log").open("a", buffering=1, encoding="utf-8") as rf:
        rf.write(f"===== [{now_str()}] RUNNER END (all_ok={ok_all}) =====\n")

    print(f"[{now_str()}] ✅ 全部实验结束（all_ok={ok_all}）。")

    # 打包
    if args.zip:
        zip_path = Path(args.zip_path) if args.zip_path else (out_root.parent / f"{out_root.name}.zip")
        make_zip(out_root, zip_path, overwrite=args.zip_overwrite)

    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
