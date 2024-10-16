# Variables
PYTHON = python
PIP = pip
SRC = main.py
TESTS = main.test.py

# Phony targets
.PHONY: test lint typecheck format help

## Show help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

## Run tests
test: ## Run tests
	$(PYTHON) $(TESTS)

## Lint code with Black
lint: ## Check code formatting with Black
	black --check $(SRC) $(TESTS)

## Run static type checks with mypy
typecheck: ## Run static type checks with mypy
	mypy $(SRC) $(TESTS)

## Format code with Black
format: ## Format code using Black
	black $(SRC) $(TESTS)
