FROM python:3.11-slim AS build-base

# Build-only toolchain for the JupyterLab extension (not kept in runtime image).
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
      > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      build \
      "jupyterlab>=4.0" \
      "hatch-jupyter-builder>=0.5"

FROM build-base AS judge-builder

WORKDIR /src

# Build torch_judge wheel.
COPY pyproject.toml setup.py README.md ./
COPY torch_judge/ ./torch_judge/
RUN python -m build --wheel --outdir /tmp/wheels .

FROM build-base AS labext-builder

# Build precompiled JupyterLab extension wheel.
# Copy manifests first so dependency install is cacheable between source edits.
WORKDIR /src/labextension
COPY labextension/.yarnrc.yml labextension/package.json labextension/pyproject.toml labextension/tsconfig.json labextension/install.json ./
# Yarn resolutions reference a local patch (license-webpack-plugin fix); it must
# exist before `jlpm install` resolves dependencies.
COPY labextension/.yarn ./.yarn
RUN jlpm install

COPY labextension/style ./style
COPY labextension/src ./src
COPY labextension/torchcode_labext ./torchcode_labext
RUN jlpm run build:prod

WORKDIR /src
RUN python -m build --wheel --outdir /tmp/wheels /src/labextension


FROM python:3.11-slim AS runtime

RUN useradd -m -u 1000 user

WORKDIR /app

COPY --from=judge-builder /tmp/wheels /tmp/wheels
COPY --from=labext-builder /tmp/wheels /tmp/wheels
# torch >=2 treats NumPy as optional; without it `import torch` prints a
# "Failed to initialize NumPy" warning and tensor<->numpy interop is disabled.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
      numpy \
      /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

COPY templates/ ./templates/
COPY solutions/ ./solutions/
COPY entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# JupyterLab custom settings: fonts, theme, Simple mode, disable news.
COPY jupyter_config/overrides.json /tmp/overrides.json
RUN LAB_DIR=$(jupyter lab path 2>/dev/null | grep "Application" | head -1 | sed 's/.*: *//') && \
    mkdir -p "$LAB_DIR/settings" && \
    cp /tmp/overrides.json "$LAB_DIR/settings/overrides.json" && \
    echo "overrides.json -> $LAB_DIR/settings/"

RUN mkdir -p /app/notebooks /app/data && \
    chown -R user:user /app

USER user

ENV PORT=7860
EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
