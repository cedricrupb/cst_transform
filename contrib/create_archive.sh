#!/bin/bash

# SPDX-FileCopyrightText: 2021 Dirk Beyer <https://www.sosy-lab.org>
#
# SPDX-License-Identifier: Apache-2.0

set -eao pipefail
# IFS=$'\t\n'

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_DIR="$SCRIPT_DIR/.."
TOOL_NAME="cst_transform"

# Create temp directory and push it
TMPDIR=$(mktemp -d)

pushd "$TMPDIR" > /dev/null
ln -s "$REPO_DIR" "repo"
mkdir $TOOL_NAME

pushd $TOOL_NAME

# TODO: maybe add README too
cp -r "$REPO_DIR/"{cst_transform/,selectors/,checkpoints/,run_selector.py,run_predict.py,LICENSE,requirements.txt} .
# Prepend run_predict - hash bang, and lib path
cp "$SCRIPT_DIR/prepend_to_run_predict" .
cat run_predict.py  >> prepend_to_run_predict 
mv prepend_to_run_predict run_predict.py
cp "$SCRIPT_DIR/prepend_to_run_predict" .
cat run_selector.py  >> prepend_to_run_predict 
mv prepend_to_run_predict run_selector.py

# Install wheel dependencies from requirements.txt
# NOTE: torch-scatter==2.0.6 requires torch, otherwise it gives error.
# Find a better way to do it.
pip3 install torch==1.8.0
pip3 wheel --wheel-dir lib -r requirements.txt

# Unzip wheel files as the whl containing a native library needs to be unzipped.
# NOTE: Not unzipping all the files as it increases the number of files too much,
# and that creates an issue in executing it with vcloud.
while read F  ; do
  unzip "lib/"$F -d lib/
  rm "lib/"$F
done <"$SCRIPT_DIR/tounzip"

popd

zip -r "$TOOL_NAME.zip" "$TOOL_NAME"
popd
mv "$TMPDIR/$TOOL_NAME.zip" ./
rm -rf $TMPDIR
