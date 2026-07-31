# Detect OS
ifeq ($(OS),Windows_NT)
    # Windows specific commands
    SEP = \
    MKDIR = if not exist $(subst /,\,$(1)) mkdir $(subst /,\,$(1))
    CP = copy /Y
    RM = del /f /q
    RMDIR = rmdir /s /q
    DETECT_DOCKER = $(shell where docker 2>nul)
    DETECT_PODMAN = $(shell where podman 2>nul)
else
    SEP = /
    MKDIR = mkdir -p
    CP = cp
    RM = rm -f
    RMDIR = rm -rf
    DETECT_DOCKER = $(shell command -v docker 2>/dev/null)
    DETECT_PODMAN = $(shell command -v podman 2>/dev/null)
endif

# Determine compose and container backends
ifneq ($(DETECT_DOCKER),)
    COMPOSE = docker compose
    CONTAINER = docker
    HAVE_COMPOSE = 1
else ifneq ($(DETECT_PODMAN),)
    COMPOSE = podman compose
    CONTAINER = podman
    HAVE_COMPOSE = 1
else
    COMPOSE = docker compose
    CONTAINER = docker
    HAVE_COMPOSE = 0
endif

# Local run command (fallback when no Docker/Podman)
LOCAL_RUN_CMD = python -m jupyter notebook --port=8888 --ip=0.0.0.0 --no-browser --notebook-dir=./notebooks

.PHONY: run run-build stop clean setup-local badges

run:
ifeq ($(HAVE_COMPOSE),1)
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
	@echo "🔥 TorchCode is running in containers!"
	@echo "   Open http://localhost:8888"
	@echo ""
else
	@echo "Docker/Podman not found, falling back to local mode."
	@echo "Starting Jupyter Notebook locally..."
	@echo "Make sure you have installed dependencies: pip install jupyter notebook"
	@echo "If notebooks are not ready, run 'make setup-local' first."
	@echo ""
	$(LOCAL_RUN_CMD)
endif

run-build:
ifeq ($(HAVE_COMPOSE),1)
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) up --build -d
	@echo ""
	@echo "🔥 TorchCode is running (local build)!"
	@echo "   Open http://localhost:8888"
	@echo ""
else
	@echo "Docker/Podman not found, falling back to local mode."
	@echo "Starting Jupyter Notebook locally..."
	@echo "Make sure you have installed dependencies: pip install jupyter notebook"
	@echo "If notebooks are not ready, run 'make setup-local' first."
	@echo ""
	$(LOCAL_RUN_CMD)
endif

stop:
ifeq ($(HAVE_COMPOSE),1)
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) down
else
	@echo "No container backend detected. Nothing to stop."
	@echo "If you have a local notebook running, press Ctrl+C in its terminal to stop it."
endif

clean:
ifeq ($(HAVE_COMPOSE),1)
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) down -v
endif
	-$(RM) data/progress.json

setup-local:
ifeq ($(OS),Windows_NT)
	@if not exist "notebooks\_original_templates" mkdir "notebooks\_original_templates"
	@for %%f in (templates\*.ipynb) do copy /Y "%%f" "notebooks\_original_templates\"
	@for %%f in (templates\*.ipynb) do copy /Y "%%f" "notebooks\"
	@for %%f in (solutions\*.ipynb) do copy /Y "%%f" "notebooks\"
else
	@mkdir -p notebooks/_original_templates
	@cp templates/*.ipynb notebooks/_original_templates/
	@cp templates/*.ipynb notebooks/
	@cp solutions/*.ipynb notebooks/
endif
	@echo "✅ Local notebooks ready in ./notebooks/"

badges:
	python scripts/add_colab_badges.py
