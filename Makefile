# =============================================================================
# dist-mfu-profiler — 一键复现入口
#
# 纪律：
#   - 账本（ledger）必须能在无卡、单进程下跑通：所有理论计算纯 CPU
#   - 实验（experiments）默认支持单进程 dry-run（world_size=1，gloo），
#     确认脚本无 bug 再上 4 卡 —— 云卡时只烧在真实实验上
#   - trace 解析有 --self-test，不依赖任何真实 trace 文件
# =============================================================================
PY     ?= python
SEED   ?= 1234
MODEL  ?= qwen2.5-7b
NPROC  ?= 4

export PYTHONPATH := .
export PYTHONUNBUFFERED := 1

.DEFAULT_GOAL := help

.PHONY: help env smoke ledger test dist-check dry-run ddp fsdp2 zero tpdp \
        parse self-test report clean distclean run-all

help:
	@echo "dist-mfu-profiler"
	@echo ""
	@echo "  make env       环境指纹（卡数/NCCL/拓扑/torch）"
	@echo "  make smoke     无卡自检：账本一致性 + trace 解析自检 + 单进程 dry-run"
	@echo "  make ledger    理论账本三张表（显存/通信/MFU 上界）      -> results/"
	@echo "  make test      pytest（多卡用例自动 skip）"
	@echo "  make dist-check 探测 NCCL/NVLink 拓扑，判断链路类型"
	@echo "  make dry-run   单进程跑一遍四组实验的训练循环（验证脚本无 bug）"
	@echo "  make ddp       torchrun --nproc_per_node=$(NPROC) experiments/run_ddp.py"
	@echo "  make fsdp2     FSDP2 全分片"
	@echo "  make zero      ZeRO-1/2/3 三档"
	@echo "  make tpdp      2-way TP x 2-way DP"
	@echo "  make parse     解析 trace（需先导出 chrome trace / nsys csv）"
	@echo "  make self-test trace 解析自检（合成数据，含已知重叠率）"
	@echo "  make report    汇总 results/*.csv 成 docs 用的 Markdown 表"
	@echo "  make run-all   一条命令跑完所有能自动跑的部分（tools/run_all.sh，约 3 秒）"
	@echo "  make clean     清缓存与中间文件（保留 results/*.csv）"
	@echo "  make distclean clean + 删除全部 results 产出"

env:
	@$(PY) -m common.env --print

smoke:
	@echo "== [1/3] 环境 =="
	@$(PY) -m common.env --print || exit 1
	@echo "== [2/3] 账本一致性（纯 CPU）=="
	@$(PY) -m pytest ledger -q || exit 1
	@echo "== [3/3] trace 解析自检 + 单进程训练循环 =="
	@$(PY) -m trace_parser.report --self-test || exit 1
	@$(PY) -m experiments.run_ddp --dry-run --steps 3 || exit 1
	@echo "smoke: PASS"

ledger:
	@mkdir -p results
	@$(PY) -m ledger.memory_ledger --model $(MODEL) --zero 1,2,3 --out results
	@$(PY) -m ledger.comm_ledger --model $(MODEL) --zero 1,2,3 --out results
	@$(PY) -m ledger.mfu_ledger --model $(MODEL) --zero 1,2,3 --out results

test:
	@$(PY) -m pytest -q

dist-check:
	@$(PY) -m common.env --topology

dry-run:
	@$(PY) -m experiments.run_ddp --dry-run --steps 5
	@$(PY) -m experiments.run_fsdp2 --dry-run --steps 5
	@$(PY) -m experiments.run_zero --dry-run --steps 5 --stages 1,2,3
	@$(PY) -m experiments.run_tp_dp --dry-run --steps 5 --tp 2 --dp 2

ddp:
	@$(PY) -m torch.distributed.run --nproc_per_node=$(NPROC) \
		-m experiments.run_ddp --steps 100 --out results

fsdp2:
	@$(PY) -m torch.distributed.run --nproc_per_node=$(NPROC) \
		-m experiments.run_fsdp2 --steps 100 --out results

zero:
	@$(PY) -m torch.distributed.run --nproc_per_node=$(NPROC) \
		-m experiments.run_zero --steps 100 --stages 1,2,3 --out results

tpdp:
	@$(PY) -m torch.distributed.run --nproc_per_node=$(NPROC) \
		-m experiments.run_tp_dp --steps 100 --tp 2 --dp 2 --out results

parse:
	@mkdir -p results
	@$(PY) -m trace_parser.parse_torch_profiler --trace traces/decode_trace.json --out results

self-test:
	@$(PY) -m trace_parser.report --self-test

report:
	@mkdir -p results
	@$(PY) -m trace_parser.report --csv results --out results

run-all:
	@bash tools/run_all.sh

clean:
	@rm -rf .pytest_cache
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name '*.pyc' -delete 2>/dev/null || true
	@rm -f results/*.tmp results/*.raw
	@echo "clean: OK（results/*.csv 保留）"

distclean: clean
	@rm -f results/*.csv results/*.md results/*.png
	@rm -rf traces profiling/raw
	@echo "distclean: OK"
