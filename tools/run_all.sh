#!/usr/bin/env bash
# tools/run_all.sh —— 一键跑完本项目所有「能自动跑」的实验。
#
# 本项目 90% 的产出是纯 CPU 的（账本三表 + trace 解析），任何机器都能一键跑完。
# 跑不出来的只有一类：4 卡实测（MFU 实测值、通信实测字节），需要 4×A100 与 40 卡时。
#
# 用法：
#   bash tools/run_all.sh                 # 账本 + trace 自检（纯 CPU，约 3 秒）
#   RUN_DRYRUN=1 bash tools/run_all.sh    # 追加四组实验的 dry-run（需 torch，验证脚本无 bug）
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
TS="$(date +%Y%m%d_%H%M%S)"
LOG="results/run_${TS}"
mkdir -p "$LOG"

step() { printf '\n=== %s ===\n' "$*"; }

step "[1/4] 环境与拓扑"
python -m common.env --print | tee "$LOG/env.txt"

step "[2/4] 显存账本（四分量 × ZeRO 档位）"
python -m ledger.memory_ledger --model qwen2.5-7b --dp 4 --zero 0,1,2,3 | tee "$LOG/memory.txt"

step "[3/4] 通信账本 + MFU 上界"
python -m ledger.comm_ledger --model qwen2.5-7b --dp 4 --zero 0,1,2,3 --link pcie4 | tee "$LOG/comm.txt"
python -m ledger.mfu_ledger  --model qwen2.5-7b --dp 4 --zero 0,3 --link nvlink | tee "$LOG/mfu.txt"

step "[4/4] trace 解析器自检（用合成数据验证解析逻辑本身）"
python -m trace_parser.report --self-test | tee "$LOG/selfcheck.txt"

if [[ "${RUN_DRYRUN:-0}" == "1" ]]; then
  step "[附加] 四组实验 dry-run（单进程，验证训练循环与落盘）"
  for e in run_ddp run_fsdp2 run_tp_dp; do
    python -m "experiments.${e}" --dry-run --steps 5 2>&1 | tail -3 | tee -a "$LOG/dryrun.txt"
  done
  python -m experiments.run_zero --dry-run --steps 5 --stages 1,2,3 2>&1 | tail -3 | tee -a "$LOG/dryrun.txt"
fi

printf '\n完成。CSV 在 results/，日志在 %s\n' "$LOG"
printf '仍然缺的只有 4 卡实测项（MFU 实测、通信实测字节）——需要 4×A100，账本给的是理论上界。\n'
