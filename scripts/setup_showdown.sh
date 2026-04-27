#!/usr/bin/env bash
# Clone smogon/pokemon-showdown, configure it for local dev (port 8000, auth off),
# and start the server. Run from the repo root or any directory — it always installs
# into a sibling directory of this repo called pokemon-showdown.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_DIR="$PARENT_DIR/pokemon-showdown"

if [ ! -d "$SERVER_DIR" ]; then
    echo "Cloning smogon/pokemon-showdown..."
    git clone https://github.com/smogon/pokemon-showdown.git "$SERVER_DIR"
fi

cd "$SERVER_DIR"

echo "Installing Node dependencies..."
npm install

# Write a minimal config: local port 8000, no login-server auth.
mkdir -p config
cat > config/config.js << 'EOF'
exports.port = 8000;
exports.bindaddress = '0.0.0.0';
exports.workers = 1;
// Allow bots to log in without a signed token (development only).
exports.noguestsecurity = true;
exports.nothrottle = true;
EOF

echo "Starting Pokémon Showdown on port 8000..."
node pokemon-showdown start --port 8000
