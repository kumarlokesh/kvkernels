#!/usr/bin/env bash
# Export the host's administrator-installed certificate authorities into
# .devcontainer/extra-ca/, so the devcontainer trusts the same TLS-inspecting
# proxies as the host. Writes nothing when the host has none.
set -euo pipefail

out_dir="$(cd "$(dirname "$0")/.." && pwd)/.devcontainer/extra-ca"
mkdir -p "$out_dir"
out="$out_dir/host.pem"
: > "$out"

case "$(uname -s)" in
  Darwin)
    security find-certificate -a -p /Library/Keychains/System.keychain >> "$out" || true
    ;;
  Linux)
    for crt in /usr/local/share/ca-certificates/*.crt /usr/local/share/ca-certificates/*/*.crt; do
      [ -e "$crt" ] && cat "$crt" >> "$out"
    done
    ;;
esac

count="$(grep -c 'BEGIN CERTIFICATE' "$out" || true)"
echo "export_host_cas: wrote ${count} certificate(s) to ${out}"
