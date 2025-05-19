#!/usr/bin/env bash
set -Eeuo pipefail

# куда ставим на Личи
LICHEE=lichee-svt
# Куда ставить на Личи — под Python 3.11:
# (обычно по умолчанию /usr/local/lib/python3.11/dist-packages
#  либо /usr/lib/python3.11/dist-packages — проверьте `python3 -m site`)
DEST_PY_PKGS="/usr/lib/python3/dist-packages"

# откуда на хосте
HOST_VENV=~/anaconda3/envs/tvm20-venv
PKG_DIR=${HOST_VENV}/lib/python3.11/site-packages
TVM_PKG=${PKG_DIR}/tvm
DISTINFO=${PKG_DIR}/tvm-0.20.0.dist-info
RUNTIME_SO=${HOST_VENV}/lib/python3.11/site-packages/tvm/libtvm_runtime.so

# 1) копируем только необходимые подпапки tvm/runtime (+ ndarray, packed_func)
echo "📦 Syncing minimal tvm package to ${LICHEE}:${DEST_PY_PKGS}"
rsync -arzv \
    --delete \
    --exclude=3rdparty \
    ${TVM_PKG} \
    root@${LICHEE}:${DEST_PY_PKGS}/

# 2) копируем dist-info, чтобы pip/python узнали версию
echo "📦 Copying dist-info"
scp -r ${DISTINFO} root@${LICHEE}:${DEST_PY_PKGS}/

# 3) копируем RISC-V libtvm_runtime.so
echo "🔄 Copying RISC-V runtime .so"
scp ${RUNTIME_SO} ${LICHEE}:/tmp/libtvm_runtime.so
ssh ${LICHEE} sudo mv /tmp/libtvm_runtime.so ${DEST_PY_PKGS}/tvm/libtvm_runtime.so

# 4) обновляем ldconfig на Личи
echo "🔧 Running ldconfig"
ssh ${LICHEE} sudo ldconfig

echo "✅ Minimal Python runtime for TVM installed on ${LICHEE}"
