#!/bin/sh
# SPDX-License-Identifier: MIT
#
# Create the virtual environment if it is missing, then run the tests.
#
# The GUI tests run on Qt's offscreen platform, so this needs no display.
# Anything the options do not claim goes on to pytest:
#
#     scripts/run-tests.sh -k graph -v

set -eu

usage() {
	cat <<'USAGE'
usage: run-tests.sh [--no-install] [pytest argument...]

  --no-install  use the virtual environment as it is
  --help        show this help and exit

Everything after the options is passed to pytest.

  VENV   virtual environment to use (default .venv)
USAGE
}

top=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$top"

install=yes
while [ $# -gt 0 ]; do
	case $1 in
	--no-install) install=no ;;
	--help) usage; exit 0 ;;
	--) shift; break ;;
	*) break ;;
	esac
	shift
done

venv=${VENV:-$top/.venv}
if [ ! -x "$venv/bin/python" ]; then
	python3 -m venv "$venv"
	install=yes
fi
if [ "$install" = yes ]; then
	"$venv/bin/pip" install --quiet --editable '.[gui,mcp,test]'
fi

exec "$venv/bin/python" -m pytest "$@"
