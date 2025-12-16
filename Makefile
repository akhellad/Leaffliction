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
        clean-pycache clean-all distribution augmentation transformation \
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
	@echo "  $(COLOR_GREEN)clean-augmented$(COLOR_RESET)      Remove augmented data directory (data/augmented_directory/)"
	@echo "  $(COLOR_GREEN)clean-transformed$(COLOR_RESET)    Remove transformed data directory (data/transformed_directory/)"
	@echo "  $(COLOR_GREEN)clean-test$(COLOR_RESET)           Remove test output directories"
	@echo "  $(COLOR_GREEN)clean-charts$(COLOR_RESET)         Remove generated charts directory"
	@echo "  $(COLOR_GREEN)clean-pycache$(COLOR_RESET)        Remove Python cache files (__pycache__, *.pyc)"
	@echo "  $(COLOR_YELLOW)clean-all$(COLOR_RESET)            Remove ALL generated data (requires confirmation)"
	@echo ""
	@echo "$(COLOR_CYAN)$(COLOR_BOLD)Execution:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)distribution$(COLOR_RESET)         Run distribution analysis on data/images/"
	@echo "  $(COLOR_GREEN)augmentation$(COLOR_RESET)         Balance/augment dataset in-place using $(AUG_SCRIPT) (AUG_PATH=$(AUG_PATH))"
	@echo "  $(COLOR_GREEN)transformation$(COLOR_RESET)       Run transformation pipeline on Apple_healthy class"
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
clean-augmented:
	@echo "$(COLOR_YELLOW)Removing augmented data directory...$(COLOR_RESET)"
	rm -rf data/augmented_directory/*
	@echo "$(COLOR_GREEN)✓ Augmented data removed$(COLOR_RESET)"

clean-transformed:
	@echo "$(COLOR_YELLOW)Removing transformed data directory...$(COLOR_RESET)"
	rm -rf data/transformed_directory/
	@echo "$(COLOR_GREEN)✓ Transformed data removed$(COLOR_RESET)"

clean-test:
	@echo "$(COLOR_YELLOW)Removing test output directories...$(COLOR_RESET)"
	rm -rf data/test_output/
	rm -rf data/test_output_full/
	rm -rf data/test_small/
	@echo "$(COLOR_GREEN)✓ Test directories removed$(COLOR_RESET)"

clean-charts:
	@echo "$(COLOR_YELLOW)Removing generated charts...$(COLOR_RESET)"
	rm -rf charts/
	@echo "$(COLOR_GREEN)✓ Charts directory removed$(COLOR_RESET)"

clean-pycache:
	@echo "$(COLOR_YELLOW)Removing Python cache files...$(COLOR_RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "$(COLOR_GREEN)✓ Python cache files removed$(COLOR_RESET)"

clean-all:
	@echo "$(COLOR_YELLOW)$(COLOR_BOLD)WARNING: This will delete ALL generated data!$(COLOR_RESET)"
	@echo "This includes:"
	@echo "  - data/augmented_directory/"
	@echo "  - data/transformed_directory/"
	@echo "  - data/test_output*, data/test_small/"
	@echo "  - charts/"
	@echo "  - __pycache__ and *.pyc files"
	@echo ""
	@printf "Are you sure you want to delete all generated data? [y/N] "; \
	read REPLY; \
	case "$$REPLY" in \
		[Yy]*) \
			$(MAKE) clean-augmented clean-transformed clean-test clean-charts clean-pycache; \
			echo "$(COLOR_GREEN)$(COLOR_BOLD)✓ All generated data removed$(COLOR_RESET)"; \
			;; \
		*) \
			echo "$(COLOR_CYAN)Cleanup cancelled$(COLOR_RESET)"; \
			;; \
	esac

# ==============================================================================
# EXECUTION TARGETS - Pipeline Scripts
# ==============================================================================
distribution:
	@echo "$(COLOR_BOLD)Running distribution analysis...$(COLOR_RESET)"
	python3 Distribution.py -src data/images
	@echo "$(COLOR_GREEN)✓ Distribution analysis complete$(COLOR_RESET)"

augmentation:
	@echo "$(COLOR_BOLD)Running data augmentation...$(COLOR_RESET)"
	$(PYTHON) $(AUG_SCRIPT) $(AUG_PATH)
	@echo "$(COLOR_GREEN)✓ Data augmentation complete$(COLOR_RESET)"

transformation:
	@echo "$(COLOR_BOLD)Running transformation pipeline (batch mode: Apple_healthy)...$(COLOR_RESET)"
	python3 Transformation.py -src data/images/Apple_healthy -dst data/transformed_directory/Apple_healthy
	@echo "$(COLOR_GREEN)✓ Transformation pipeline complete$(COLOR_RESET)"

pipeline:
	@echo "$(COLOR_MAGENTA)$(COLOR_BOLD)Starting full pipeline execution...$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_CYAN)Step 1/3: Distribution Analysis$(COLOR_RESET)"
	@$(MAKE) distribution
	@echo ""
	@echo "$(COLOR_CYAN)Step 2/3: Data Augmentation$(COLOR_RESET)"
	@$(MAKE) augmentation
	@echo ""
	@echo "$(COLOR_CYAN)Step 3/3: Transformation Pipeline$(COLOR_RESET)"
	@$(MAKE) transformation
	@echo ""
	@echo "$(COLOR_MAGENTA)$(COLOR_BOLD)✓ Full pipeline execution complete!$(COLOR_RESET)"

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
