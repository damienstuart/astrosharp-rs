# Common tasks for astrosharp-rs. Run `make` or `make help` for the list.

CARGO   ?= cargo
# The original R app, needed only for `golden` and `weights`.
R_APP   ?= ../AstroSharp/resources/app
GOLDEN  := testdata/golden
APP     := target/release/bundle/AstroSharp.app
APP_DIR ?= /Applications

.DEFAULT_GOAL := help
.PHONY: help build release test check lint fmt run cli app install golden weights clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Debug build of all crates
	$(CARGO) build --workspace

release: ## Optimised build of the CLI (astrosharp) and GUI (astrosharp-gui)
	$(CARGO) build --workspace --release

test: ## Run all tests (R parity tests are skipped until `make golden` has run)
	$(CARGO) test --workspace --release

lint: ## Check formatting and run clippy (warnings are errors)
	$(CARGO) fmt --all --check
	$(CARGO) clippy --workspace --all-targets --release -- -D warnings

check: lint test ## lint + test

fmt: ## Format all code
	$(CARGO) fmt --all

run: ## Run the GUI; open an image with `make run IMAGE=path.tif`
	$(CARGO) run --release -p astrosharp-gui -- $(IMAGE)

cli: release ## Build the CLI and show its options
	./target/release/astrosharp --help

app: ## Build an unsigned macOS app bundle in target/release/bundle/
	./tools/bundle_macos.sh

install: app ## Copy AstroSharp.app to /Applications (override with APP_DIR=...)
	rm -rf "$(APP_DIR)/AstroSharp.app"
	cp -R "$(APP)" "$(APP_DIR)/"
	@echo "Installed $(APP_DIR)/AstroSharp.app"

golden: ## Regenerate R reference outputs (needs R and the original app at R_APP)
	Rscript tools/make_golden.R "$(R_APP)" $(GOLDEN)

weights: ## Re-export model weights from the R app's .RDS files
	Rscript tools/export_weights.R "$(R_APP)" crates/core/weights

clean: ## Remove build artifacts
	$(CARGO) clean
