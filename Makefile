COMPOSE ?= $(shell \
	if command -v docker >/dev/null 2>&1; then \
		echo "docker compose"; \
	elif command -v podman >/dev/null 2>&1; then \
		echo "podman compose"; \
	else \
		echo "docker compose"; \
	fi)
CONTAINER ?= $(shell \
	if command -v docker >/dev/null 2>&1; then \
		echo "docker"; \
	elif command -v podman >/dev/null 2>&1; then \
		echo "podman"; \
	else \
		echo "docker"; \
	fi)

.PHONY: run run-build stop clean setup-local badges

run:
	@echo "Using compose backend: $(COMPOSE)"
	@echo "Pulling prebuilt image (if available)..."
	@pull_status=0; \
	images="$$( $(COMPOSE) config --images 2>/dev/null )"; \
	$(COMPOSE) pull || pull_status=$$?; \
	missing_images=""; \
	for image in $$images; do \
		if ! $(CONTAINER) image inspect "$$image" >/dev/null 2>&1; then \
			missing_images="$$missing_images $$image"; \
		fi; \
	done; \
	if [ -n "$$images" ] && [ -z "$$missing_images" ]; then \
		if [ $$pull_status -ne 0 ]; then \
			echo "Prebuilt image pull failed, but local compose images are available. Reusing them."; \
		fi; \
		$(COMPOSE) up -d --no-build; \
	else \
		echo "Prebuilt image unavailable for this platform. Building locally instead..."; \
		$(COMPOSE) up --build -d; \
	fi
	@echo ""
	@echo "🔥 TorchCode is running!"
	@echo "   Open http://localhost:8888"
	@echo ""

run-build:
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) up --build -d
	@echo ""
	@echo "🔥 TorchCode is running (local build)!"
	@echo "   Open http://localhost:8888"
	@echo ""

stop:
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) down

clean:
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) down -v
	rm -f data/progress.json

setup-local:
	@mkdir -p notebooks/_original_templates
	@cp templates/*.ipynb notebooks/_original_templates/
	@cp templates/*.ipynb notebooks/
	@cp solutions/*.ipynb notebooks/
	@echo "✅ Local notebooks ready in ./notebooks/"

badges:
	python scripts/add_colab_badges.py
