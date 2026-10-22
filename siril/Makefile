# Tasks for the AstroSharp Siril script. Run `make` or `make help` for the list.

PYTHON        ?= python3
VENV          := .venv
SIRIL_SCRIPTS ?= $(HOME)/.siril/scripts
# For the CLI integration test (skipped if missing).
ASTROSHARP_CLI  ?= /Applications/AstroSharp.app/Contents/MacOS/astrosharp-cli
ASTROSHARP_FITS ?=

.DEFAULT_GOAL := help
.PHONY: help test install clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-z-]+:.*## / {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --quiet numpy PyQt6  # PyQt6: window smoke test only

test: $(VENV)/bin/python ## Run the tests (set ASTROSHARP_FITS=file.fit for the CLI integration test)
	ASTROSHARP_CLI="$(ASTROSHARP_CLI)" ASTROSHARP_FITS="$(ASTROSHARP_FITS)" \
		$(VENV)/bin/python -m unittest discover -s tests -v

install: ## Copy AstroSharp.py to Siril's scripts folder (SIRIL_SCRIPTS=... to change)
	mkdir -p "$(SIRIL_SCRIPTS)"
	cp AstroSharp.py "$(SIRIL_SCRIPTS)/"
	@echo "Installed $(SIRIL_SCRIPTS)/AstroSharp.py (refresh Siril's script list to see it)"

clean: ## Remove the test virtualenv
	rm -rf $(VENV)
