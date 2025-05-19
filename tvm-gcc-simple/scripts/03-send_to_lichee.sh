#!/usr/bin/env bash
set -Ee

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

rsync -arzP "${SCRIPT_DIR}"/../data/tvm/dist/riscv64/tvm-0.20.0-py2.py3-none-any.whl lichee-svt:/home/sipeed/TVM/
rsync -arzP "${SCRIPT_DIR}"/model.so lichee-svt:/home/sipeed/TVM/
rsync -arzP "${SCRIPT_DIR}"/model.json lichee-svt:/home/sipeed/TVM/
rsync -arzP "${SCRIPT_DIR}"/model.params lichee-svt:/home/sipeed/TVM/
rsync -arzP "${SCRIPT_DIR}"/test_tvm_runtime.py lichee-svt:/home/sipeed/TVM/

