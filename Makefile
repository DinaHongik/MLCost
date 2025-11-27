# MLCost - ML Training Cost Predictor

IMAGE_NAME = mlcost
CONTAINER_NAME = mlcost-container
PYTHON = python3

DATA_DIR = ./data
RAW_DIR = $(DATA_DIR)/raw
EXTRACTED_DIR = $(DATA_DIR)/extracted
PROCESSED_DIR = $(DATA_DIR)/processed
MODELS_DIR = ./models
EXPERIMENTS_DIR = ./experiments

.DEFAULT_GOAL := help

# =============================================================================
# Help
# =============================================================================

.PHONY: help
help:
	@echo "MLCost - ML Training Cost Predictor"
	@echo ""
	@echo "Quick Start:"
	@echo "  make all              - Full pipeline (download → predict)"
	@echo ""
	@echo "Docker:"
	@echo "  make build            - Build Docker image"
	@echo "  make shell            - Interactive shell"
	@echo "  make stop             - Stop container"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make download         - Clone MLPerf repositories"
	@echo "  make extract          - Parse logs to CSV"
	@echo "  make preprocess       - Generate features (420D)"
	@echo ""
	@echo "Model:"
	@echo "  make train            - Train ensemble models"
	@echo "  make experiments      - Run paper experiments"
	@echo ""
	@echo "Prediction:"
	@echo "  make predict Q=\"...\"  - Single prediction"
	@echo "  make interactive      - Interactive mode"
	@echo ""
	@echo "Utilities:"
	@echo "  make status           - Show pipeline status"
	@echo "  make clean            - Remove generated files"

# =============================================================================
# Docker
# =============================================================================

.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

.PHONY: shell
shell:
	docker run --gpus all --rm -it \
		-v $(shell pwd):/workspace \
		-w /workspace \
		$(IMAGE_NAME) bash

.PHONY: start
start:
	docker run --gpus all -d --name $(CONTAINER_NAME) \
		-v $(shell pwd):/workspace \
		-w /workspace \
		$(IMAGE_NAME)
	@echo "Container started. Use 'make exec' to attach."

.PHONY: exec
exec:
	docker exec -it $(CONTAINER_NAME) bash

.PHONY: stop
stop:
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true

# =============================================================================
# Data Pipeline
# =============================================================================

.PHONY: download
download:
	@mkdir -p $(RAW_DIR)
	$(PYTHON) scripts/downloader.py --config config/config.json

.PHONY: extract
extract:
	@mkdir -p $(EXTRACTED_DIR)
	$(PYTHON) scripts/extractor.py \
		--input $(RAW_DIR) \
		--output $(EXTRACTED_DIR)/mlperf_results.csv

.PHONY: preprocess
preprocess:
	@mkdir -p $(PROCESSED_DIR)
	$(PYTHON) mlcost/preprocess.py

# =============================================================================
# Model
# =============================================================================

.PHONY: train
train:
	$(PYTHON) mlcost/predictor.py --retrain

.PHONY: experiments
experiments:
	$(PYTHON) mlcost/experiments.py

# =============================================================================
# Prediction
# =============================================================================

.PHONY: predict
predict:
ifndef Q
	@echo "Usage: make predict Q=\"BERT on 8 A100 GPUs\""
	@exit 1
endif
	$(PYTHON) mlcost/predictor.py --query "$(Q)"

.PHONY: interactive
interactive:
	$(PYTHON) mlcost/predictor.py

# =============================================================================
# Full Pipeline
# =============================================================================

.PHONY: all
all: download extract preprocess train
	@echo "Pipeline complete!"

# =============================================================================
# Utilities
# =============================================================================

.PHONY: status
status:
	@echo "=== MLCost Status ==="
	@echo ""
	@echo "Data:"
	@[ -d "$(RAW_DIR)" ] && echo "  [x] Raw data" || echo "  [ ] Raw data"
	@[ -f "$(EXTRACTED_DIR)/mlperf_results.csv" ] && echo "  [x] Extracted" || echo "  [ ] Extracted"
	@[ -d "$(PROCESSED_DIR)/swmf_numeric_faiss" ] && echo "  [x] Processed" || echo "  [ ] Processed"
	@echo ""
	@echo "Models:"
	@[ -f "$(MODELS_DIR)/rf_model.pkl" ] && echo "  [x] Trained" || echo "  [ ] Trained"

.PHONY: clean
clean:
	rm -rf $(PROCESSED_DIR)
	rm -rf $(MODELS_DIR)
	rm -rf $(EXPERIMENTS_DIR)
	rm -rf __pycache__ mlcost/__pycache__