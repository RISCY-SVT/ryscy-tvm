#!/usr/bin/env bash
set -Ee

# source ./env.sh

# scp $Dev_BUILD/libtvm_runtime.so            lichee-svt:/usr/local/lib/
# scp -r $TVM_HOME/include                    lichee-svt:/usr/local/include/tvm
# scp ./build_th1520/yolov5n_static_th1520.so lichee-svt:/home/sipeed/TVM/
# ssh lichee-svt 'echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/tvm.conf && sudo ldconfig'

rsync -arzP /home/svt/Custler/TVM/tvm-v0.20.0/dist/riscv64/tvm-0.20.0-py2.py3-none-any.whl lichee-svt:/home/sipeed/TVM/
rsync -arzP /home/svt/Custler/TVM/model.so lichee-svt:/home/sipeed/TVM/
rsync -arzP /home/svt/Custler/TVM/model.json lichee-svt:/home/sipeed/TVM/
rsync -arzP /home/svt/Custler/TVM/model.params lichee-svt:/home/sipeed/TVM/
rsync -arzP /home/svt/Custler/TVM/test_tvm_runtime.py lichee-svt:/home/sipeed/TVM/

