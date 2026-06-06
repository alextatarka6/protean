#!/usr/bin/env bash
# Start the protean local Showdown server (port 8001).
# Usage:
#   ./scripts/start_server.sh          # foreground (Ctrl-C to stop)
#   ./scripts/start_server.sh &        # background

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/../server/pokemon-showdown"

echo "Starting Pokémon Showdown server on port 8001..."
cd "$SERVER_DIR"
exec node pokemon-showdown start --no-security
