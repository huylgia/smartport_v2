.DEFAULT_GOAL := help
UV ?= uv

# Cổng Triton lấy từ build/.env.triton — máy dùng chung nên cổng phải cấu hình được.
COMPOSE := docker compose --env-file build/.env.triton -f build/docker-compose.triton.yml
HTTP_PORT = $(shell grep TRITON_HTTP_PORT build/.env.triton | cut -d= -f2)
GRPC_PORT = $(shell grep TRITON_GRPC_PORT build/.env.triton | cut -d= -f2)
METRICS_PORT = $(shell grep TRITON_METRICS_PORT build/.env.triton | cut -d= -f2)


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

# ---------------------------------------------------------------- Triton
.PHONY: build-triton
build-triton: ## Docker image cho Triton + modelsvc
	docker build -t craneops-triton:dev -f build/triton.Dockerfile .

.PHONY: up
up: ## Dựng engine (nếu cần) rồi chạy Triton
	$(COMPOSE) up -d
	@echo "Triton: HTTP :$(HTTP_PORT)  gRPC :$(GRPC_PORT)  metrics :$(METRICS_PORT)"

.PHONY: down
down: ## Dừng Triton. Engine nằm trong volume `craneops_models`, KHÔNG mất.
	$(COMPOSE) down

.PHONY: status
status: ## Model nào đang READY
	@curl -s -X POST localhost:$(HTTP_PORT)/v2/repository/index | \
	 python3 -c "import json,sys;[print(f\"  {m['name']:<28}{m['state']}\") for m in sorted(json.load(sys.stdin),key=lambda x:x['name'])]"

.PHONY: logs
logs: ## Log của modelsvc (dựng engine) và Triton
	$(COMPOSE) logs --tail 40

# ---------------------------------------------------------------- đo đạc
.PHONY: bench
bench: ## Hiệu năng: từng model + đường ống BLS (HARDWARE_BUDGET §6.1)
	$(UV) run --with "tritonclient[grpc]" python -m tools.bench.triton_bench --all \
		--http http://localhost:$(HTTP_PORT) --url localhost:$(GRPC_PORT) \
		--metrics http://localhost:$(METRICS_PORT)/metrics

.PHONY: accuracy
accuracy: ## Độ chính xác từng model so với NHÃN (HARDWARE_BUDGET §6.2)
	$(UV) run --with "tritonclient[http]" --with opencv-python-headless \
		python -m tools.golden.accuracy --url localhost:$(HTTP_PORT)

.PHONY: gpu-watch
gpu-watch: ## Quan sát VRAM/NVDEC/NVENC — NVENC PHẢI bằng 0 khi ghi hình
	nvidia-smi dmon -s pucm

.PHONY: help
help:
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- ds_app -------------------------------------------------------------------
DS_COMPOSE := docker compose --env-file build/.env.ds -f build/docker-compose.ds.yml
CAM ?= 1
DUR ?= 60

.PHONY: ds-build
ds-build: ## Dựng image ds_app (DeepStream + pyds)
	$(DS_COMPOSE) build record

.PHONY: ds-doctor
ds-doctor: ## Kiểm môi trường ds_app — chạy TRƯỚC khi nghi ngờ thứ khác
	$(DS_COMPOSE) run --rm doctor

.PHONY: record
record: ## Ghi hình một camera, KHÔNG suy luận (CAM=1 DUR=60 META=1)
	CAM=$(CAM) DUR=$(DUR) META=$(META) $(DS_COMPOSE) run --rm record

.PHONY: record-clean
record-clean: ## Xoá segment đã ghi. Chạy trong container vì file thuộc root.
	$(DS_COMPOSE) run --rm --entrypoint sh record -c 'rm -rf /rec/*'
