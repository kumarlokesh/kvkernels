# Extra certificate authorities

The devcontainer image trusts every `*.pem` and `*.crt` file in this directory.

`scripts/export_host_cas.sh` runs on the host before each build (the
devcontainer's `initializeCommand`) and writes the host's administrator-installed
certificate authorities here. This lets the container reach PyPI and
download.pytorch.org through a TLS-inspecting proxy that the host already trusts.

Certificate files here are ignored by git and never committed.
