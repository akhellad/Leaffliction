# Makefile for Leaffliction Plant Disease Classification Project
# Author: Generated for leaffliction project
# Date: November 27, 2025

.DEFAULT_GOAL := help

# Color codes for formatted output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_CYAN := \033[36m
COLOR_MAGENTA := \033[35m

# Configurable paths/commands
PYTHON ?= python3
AUG_SCRIPT := scripts/Augmentation.py
# Override with `make augmentation AUG_PATH=path/to/data`
AUG_PATH ?= data/images

# PHONY declarations - all targets are commands, not files
.PHONY: help  clean-augmented clean-transformed clean-test clean-charts \
        clean-pycache clean-all distribution distribution-apple distribution-grape \
        augmentation transformation transformation-apple transformation-grape \
        pipeline lint train predict

# ==============================================================================
# HELP TARGET - Display all available commands
# ==============================================================================
help:
	@echo "$(COLOR_BOLD)Leaffliction Makefile - Plant Disease Classification Pipeline$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Usage:$(COLOR_RESET) make [target]"
	@echo ""
	@echo "$(COLOR_CYAN)$(COLOR_BOLD)Setup:$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_CYAN)$(COLOR_BOLD)Cleaning:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)clean-pycache$(COLOR_RESET)        Remove Python cache files (__pycache__, *.pyc)"
	@echo ""
	@echo "$(COLOR_CYAN)$(COLOR_BOLD)Execution:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)distribution-apple$(COLOR_RESET)   Run distribution analysis on Apple/ directory"
	@echo "  $(COLOR_GREEN)distribution-grape$(COLOR_RESET)   Run distribution analysis on Grape/ directory"
	@echo "  $(COLOR_GREEN)augmentation$(COLOR_RESET)         Balance/augment dataset in-place using $(AUG_SCRIPT) (AUG_PATH=$(AUG_PATH))"
	@echo "  $(COLOR_GREEN)transformation-apple$(COLOR_RESET) Apply transformations to all Apple disease subdirectories"
	@echo "  $(COLOR_GREEN)transformation-grape$(COLOR_RESET) Apply transformations to all Grape disease subdirectories"
	@echo "  $(COLOR_MAGENTA)pipeline$(COLOR_RESET)             Run full pipeline (distribution → augmentation → transformation)"
	@echo ""
	@echo "$(COLOR_CYAN)$(COLOR_BOLD)Code Quality:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)lint$(COLOR_RESET)                 Run flake8 linter (79 char limit, PEP8 compliance)"
	@echo ""
	@echo "$(COLOR_CYAN)$(COLOR_BOLD)Future Targets:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)train$(COLOR_RESET)                Train CNN model (not yet implemented)"
	@echo "  $(COLOR_GREEN)predict$(COLOR_RESET)              Run inference on image (not yet implemented)"
	@echo ""


# ==============================================================================
# CLEANING TARGETS
# ==============================================================================

clean-pycache:
	@echo "$(COLOR_YELLOW)Removing Python cache files...$(COLOR_RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(COLOR_GREEN)✓ Python cache files removed$(COLOR_RESET)"

# ==============================================================================
# EXECUTION TARGETS - Pipeline Scripts
# ==============================================================================
distribution-apple:
	@echo "$(COLOR_BOLD)Running distribution analysis for Apple...$(COLOR_RESET)"
	$(PYTHON) scripts/Distribution.py Apple/
	@echo "$(COLOR_GREEN)✓ Apple distribution analysis complete$(COLOR_RESET)"

distribution-grape:
	@echo "$(COLOR_BOLD)Running distribution analysis for Grape...$(COLOR_RESET)"
	$(PYTHON) scripts/Distribution.py Grape/
	@echo "$(COLOR_GREEN)✓ Grape distribution analysis complete$(COLOR_RESET)"

augmentation:
	@echo "$(COLOR_BOLD)Running data augmentation...$(COLOR_RESET)"
	$(PYTHON) $(AUG_SCRIPT) $(AUG_PATH)
	@echo "$(COLOR_GREEN)✓ Data augmentation complete$(COLOR_RESET)"

transformation-apple:
	@echo "$(COLOR_BOLD)Running transformation pipeline for Apple...$(COLOR_RESET)"
	@mkdir -p transformed/Apple_Black_rot transformed/Apple_healthy transformed/Apple_rust transformed/Apple_scab
	$(PYTHON) scripts/Transformation.py -src Apple/Apple_Black_rot -dst transformed/Apple_Black_rot
	$(PYTHON) scripts/Transformation.py -src Apple/Apple_healthy -dst transformed/Apple_healthy
	$(PYTHON) scripts/Transformation.py -src Apple/Apple_rust -dst transformed/Apple_rust
	$(PYTHON) scripts/Transformation.py -src Apple/Apple_scab -dst transformed/Apple_scab
	@echo "$(COLOR_GREEN)✓ Apple transformation pipeline complete$(COLOR_RESET)"

transformation-grape:
	@echo "$(COLOR_BOLD)Running transformation pipeline for Grape...$(COLOR_RESET)"
	@mkdir -p transformed/Grape_Black_rot transformed/Grape_Esca transformed/Grape_healthy transformed/Grape_spot
	$(PYTHON) scripts/Transformation.py -src Grape/Grape_Black_rot -dst transformed/Grape_Black_rot
	$(PYTHON) scripts/Transformation.py -src Grape/Grape_Esca -dst transformed/Grape_Esca
	$(PYTHON) scripts/Transformation.py -src Grape/Grape_healthy -dst transformed/Grape_healthy
	$(PYTHON) scripts/Transformation.py -src Grape/Grape_spot -dst transformed/Grape_spot
	@echo "$(COLOR_GREEN)✓ Grape transformation pipeline complete$(COLOR_RESET)"

# ==============================================================================
# CODE QUALITY TARGETS
# ==============================================================================
lint:
	@echo "$(COLOR_BOLD)Running flake8 linter (PEP8 compliance check)...$(COLOR_RESET)"
	python -m flake8 *.py src/ --max-line-length=79 --show-source --statistics
	@echo "$(COLOR_GREEN)✓ Lint check complete$(COLOR_RESET)"

# ==============================================================================
# FUTURE TARGETS - Placeholders for upcoming features
# ==============================================================================
train:
	@echo "$(COLOR_YELLOW)$(COLOR_BOLD)[TODO]$(COLOR_RESET) train.py not yet implemented."
	@echo "See .github/copilot-instructions.md for specifications."
	@echo ""
	@echo "Expected usage:"
	@echo "  python train.py --source data/augmented_directory --epochs 50 --batch-size 32"

predict:
	@echo "$(COLOR_YELLOW)$(COLOR_BOLD)[TODO]$(COLOR_RESET) predict.py not yet implemented."
	@echo "See .github/copilot-instructions.md for specifications."
	@echo ""
	@echo "Expected usage:"
	@echo "  python predict.py --image path/to/leaf.jpg --model models/best_model.h5"
