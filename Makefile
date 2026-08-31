.DEFAULT_GOAL := help
UV ?= uv



# ---------------------------------------------------------------- dev
.PHONY: setup
setup: ## Cài dependency + pre-commit hook
	$(UV) sync --extra dev
	$(UV) run pre-commit install

.PHONY: lint
lint: ## ruff check + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

.PHONY: fmt
fmt: ## ruff --fix + format
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

.PHONY: type
type: ## mypy
	$(UV) run mypy .

.PHONY: test
test: ## unit test + coverage
	$(UV) run pytest tests/unit --cov --cov-report=term-missing

.PHONY: check
check: lint type test ## Tất cả kiểm tra chạy trên CI

.PHONY: schema
schema: ## Sinh lại ví dụ message trong docs/MESSAGE_CONTRACT.md
	$(UV) run python -m tools.gen_message_examples

# ---------------------------------------------------------------- model
.PHONY: fold
fold: ## Gấp tiền xử lý vào đồ thị ONNX -> *_folded.t7 (DN-012)
	$(UV) run --extra export --with onnxruntime python -m tools.fold_preprocess

.PHONY: fold-check
fold-check: ## Kiểm chứng các *_folded.t7 đã sinh, không ghi đè
	$(UV) run --extra export --with onnxruntime python -m tools.fold_preprocess --check

.PHONY: config
config: ## Sinh lại triton/repo/**/config.pbtxt từ SPECS
	$(UV) run python -m tools.export_models --emit-config

.PHONY: config-check
config-check: ## CI: fail nếu config.pbtxt trôi khỏi SPECS
	$(UV) run python -m tools.export_models --check


# ---------------------------------------------------------------- vận hành
# Vận hành đã chuyển sang CLI trong `deploy/` — nó chạy được trên máy đích, nơi chỉ có
# Docker chứ không có venv hay `uv`. Các target dưới đây chỉ là lối tắt cho máy dev; MỘT
# bản cài đặt duy nhất, nên không thể trôi khỏi nhau.
#
#   ./deploy/craneops build           dựng image mọi service
#   ./deploy/craneops-triton up       chỉ Triton
#   ./deploy/craneops-ds record       ghi hình
#   ./deploy/craneops services        xem toàn bộ lệnh

.PHONY: up down status logs bench accuracy
up down status logs bench accuracy: ## → deploy/craneops-triton <lệnh>
	@./deploy/craneops-triton $@

.PHONY: ds-build ds-doctor
ds-build:  ## → deploy/craneops-ds build
	@./deploy/craneops-ds build
ds-doctor: ## → deploy/craneops-ds doctor
	@./deploy/craneops-ds doctor

.PHONY: build-triton
build-triton: ## → deploy/craneops-triton build
	@./deploy/craneops-triton build

# Mặc định của `make record`. CLI có mặc định riêng; hai chỗ này chỉ để `make record`
# không truyền cờ rỗng.
CAM ?= 1
DUR ?= 60

.PHONY: record record-clean
record: ## → deploy/craneops-ds record (CAM=1 DUR=60 SEGMENT_SEC=10 META=1)
	@./deploy/craneops-ds record --cam $(CAM) --duration $(DUR) \
	  $(if $(SEGMENT_SEC),--segment-sec $(SEGMENT_SEC),) $(if $(META),--meta,)
record-clean: ## → deploy/craneops-ds clean
	@./deploy/craneops-ds clean

.PHONY: gpu-watch
gpu-watch: ## Quan sát VRAM/NVDEC/NVENC — NVENC PHẢI bằng 0 khi ghi hình
	nvidia-smi dmon -s pucm

.PHONY: help
help:
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
