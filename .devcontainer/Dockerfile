# CPU-only Linux environment for running tests under the Triton interpreter.
FROM python:3.12-slim

# Trust the certificate authorities exported from the host by
# scripts/export_host_cas.sh, so TLS-inspecting proxies work inside the container.
COPY extra-ca/ /tmp/extra-ca/
RUN set -eux; \
    mkdir -p /usr/local/share/ca-certificates/extra; \
    for pem in /tmp/extra-ca/*.pem /tmp/extra-ca/*.crt; do \
        [ -e "$pem" ] || continue; \
        name="$(basename "$pem")"; \
        awk -v out="/usr/local/share/ca-certificates/extra/${name%.*}" \
            '/BEGIN CERTIFICATE/ { n++ } n { print > (out "-" n ".crt") }' "$pem"; \
    done; \
    update-ca-certificates; \
    rm -rf /tmp/extra-ca

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends git make; \
    rm -rf /var/lib/apt/lists/*; \
    pip install --no-cache-dir uv==0.13.5

# Keep the Linux virtual environment out of the workspace, which the host shares.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy
