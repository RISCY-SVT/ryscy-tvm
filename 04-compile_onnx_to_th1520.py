#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04-compile_onnx_to_th1520.py

ONNX → TVM Relax (v0.20) → RISC-V RVV (Clang/LLVM) shared library.
Перед линковкой сохраняем сгенерированный C-код.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import onnx
import numpy as np
import tvm
from tvm import relax
from tvm.driver import build_module
from tvm.driver.build_module import compile as tvm_compile
from tvm.contrib import cc
from onnx import helper, numpy_helper

# ------------------------------------------------------------------------
def tensorize_all_scalars(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    В OLDER ONNX: заменяем все скалярные Constant (value_float/int / value_floats/ints len==1)
    на rank-1 тензор shape=[1], чтобы Relax.floor() и др. видели Tensor, а не PrimValue.
    """
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        to_remove = []
        new_tensor = None
        for attr in list(node.attribute):
            # float/int scalar
            if attr.name == "value_float":
                arr = np.array([attr.f], dtype="float32")
                new_tensor = numpy_helper.from_array(arr)
                to_remove.append(attr)
            elif attr.name == "value_int":
                arr = np.array([attr.i], dtype="int64")
                new_tensor = numpy_helper.from_array(arr)
                to_remove.append(attr)
            # list of floats/ints, length==1
            elif attr.name == "value_floats" and len(attr.floats) == 1:
                arr = np.array([attr.floats[0]], dtype="float32")
                new_tensor = numpy_helper.from_array(arr)
                to_remove.append(attr)
            elif attr.name == "value_ints" and len(attr.ints) == 1:
                arr = np.array([attr.ints[0]], dtype="int64")
                new_tensor = numpy_helper.from_array(arr)
                to_remove.append(attr)
            # already TensorProto rank-0
            elif attr.name == "value" and attr.t.dims == []:
                arr = numpy_helper.to_array(attr.t)
                arr = arr.reshape(1)
                new_tensor = numpy_helper.from_array(arr)
                to_remove.append(attr)
        if new_tensor is not None:
            # удалить старые атрибуты
            for a in to_remove:
                node.attribute.remove(a)
            # добавить новый
            node.attribute.extend([helper.make_attribute("value", new_tensor)])
    onnx.checker.check_model(model)
    return model

# ------------------------------------------------------------------------
def get_args():
    P = argparse.ArgumentParser(
        description="Compile ONNX → Relax (TVM v0.20) → RISC-V RVV via Clang/LLVM"
    )
    P.add_argument("model", help="путь к ONNX-модели (static/dynamic)")
    P.add_argument("--input-name", default="images")
    P.add_argument("--input-shape", nargs=4, type=int,
                   default=[1,3,640,640], metavar=("N","C","H","W"))
    P.add_argument("--dtype",    default="float32")
    P.add_argument("--out-dir",  default="build_th1520")
    return P.parse_args()

# ------------------------------------------------------------------------
def main():
    args = get_args()
    onnx_path = pathlib.Path(args.model).resolve()
    out_dir   = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"⏳  Loading ONNX model: {onnx_path.name}")
    model = onnx.load(str(onnx_path))
    print("🔧  Tensorizing all scalar Constants …")
    model = tensorize_all_scalars(model)

    # ------------------- ONNX → Relax -------------------
    from tvm.relax.frontend.onnx import from_onnx
    sd = {args.input_name: tuple(args.input_shape)}
    dd = {args.input_name: args.dtype}

    print("🔄  Converting ONNX → Relax …")
    mod = from_onnx(model, shape_dict=sd, dtype_dict=dd)
    mod, params = relax.frontend.detach_params(mod)

    # ------------------- Выбор таргета: LLVM/Clang для RISC-V RVV -------------
    # используем YOUR LLVM-RVV-Clang 16.0.0 в /opt/riscv/bin
    target = tvm.target.Target(
        "llvm"
        " -mtriple=riscv64-unknown-linux-gnu"
        " -mcpu=generic-rv64"
        " -mattr=+m,+v,+zfh"
    )

    print("⚙️  Compiling Relax → Executable …")
    exe = tvm_compile(
        mod,
        target,                         # ваш Target
        relax_pipeline="default",      # pipeline по умолчанию для Relax
        tir_pipeline="default"         # pipeline по умолчанию для TIR
    )

    # ------------------- Экспорт .so с предсохранением C-кода ----------------
    stem = onnx_path.stem + "_th1520"
    so_path = out_dir / (stem + ".so")

    # флаги для Clang+LLD
    clang = "/opt/riscv/bin/clang"
    llvm_config = "/opt/riscv/bin/llvm-config"
    # include от TVM и зависимостей:
    tvm_home = os.environ.get("TVM_HOME", "")
    SYSROOT = "/opt/riscv/sysroot"
    
    # Пути включений TVM
    incs = [
        f"-I{tvm_home}/include",
        f"-I{tvm_home}/3rdparty/dlpack/include",
        f"-I{tvm_home}/3rdparty/dmlc-core/include",
    ]
    
    # Собираем опции для Clang+LLD
    # Путь к CRT вашей версии ISA
    crt_dir = f"{SYSROOT}/usr/lib64v0p7_xthead/lp64d"
    
    clang_opts = [
        "--target=riscv64-unknown-linux-gnu",
        f"--sysroot={SYSROOT}",
        "-nostdlib",                                # сброс поиска CRT вне sysroot
        f"-B{crt_dir}",                             # искать CRT именно здесь
        "-fPIC", "-shared",
        "-march=rv64gcv_zfh_xtheadc",
        "-mabi=lp64d",
        "-fuse-ld=lld",
        # explicit CRT order:
        f"{crt_dir}/crti.o",
        # библиотечные пути
        f"-L{tvm_home}/build-riscv",
        "-ltvm_runtime",
        # ваши object files (lib0.o, devc.o) добавятся автоматически
        # в конце CRT конца
        f"{crt_dir}/crtn.o",
    ] + incs
    
    # clang_opts = []
    # for d in inc_dirs:
    #     clang_opts.append(f"-I{d}")                        # заголовки TVM :contentReference[oaicite:5]{index=5}
    # clang_opts += [
    #     "--target=riscv64-unknown-linux-gnu",               # трипл для RISCV :contentReference[oaicite:6]{index=6}
    #     "-march=rv64gcv_zfh",                            # RVV 0.7 + zfh + xtheadc :contentReference[oaicite:7]{index=7}
    #     f"--sysroot=/opt/riscv/sysroot",                   # где искать libc, libgcc и пр.  :contentReference[oaicite:1]{index=1}
    #     f"--gcc-toolchain=/opt/riscv",                    # указываем gcc-toolchain для LLD  :contentReference[oaicite:2]{index=2}
    #     "-mabi=lp64d",                                      # ABI
    #     "-fPIC",                                            # позиционно-независимый код
    #     "-shared",                                          # создаём .so
    #     "-v",
    #     "-fuse-ld=lld",                                     # используем LLD, умеющий RVV1.0 атрибуты :contentReference[oaicite:8]{index=8}
    #     f"-L{tvm_home}/build-riscv",                        # путь к libtvm_runtime.so
    #     "-ltvm_runtime",                                    # правильный суффикс для рантайма :contentReference[oaicite:9]{index=9}
    #     # при необходимости добавьте:
    #     # "-Wl,-rpath,$ORIGIN",                             # чтобы .so искал рутаймы рядом
    # ]

    # clang_opts = []
    # for d in inc_dirs:
    #     clang_opts.append(f"-I{d}")
    # clang_opts += [
    #     "-target", "riscv64-unknown-linux-gnu",    # clang-specific
    #     "-march=rv64gcmv0p7_zfh",
    #     "-mabi=lp64d",
    #     "-fPIC",
    #     "-shared",
    #     "-v",
    #     f"-L{os.path.join(tvm_home, 'build-riscv')}",
    #     "-ltvm_runtime"
    # ]

    def my_fcompile(lib_name, object_files, **kwargs):
        """
        Сохраняем сгенерированный C-код (.c) и потом линкуем Clang.
        object_files обычно: ['/tmp/.../lib0.c', '/tmp/.../devc.o']
        """
        # ищем lib0.c (основной С-код)
        print(f"Looking fo .c in {object_files}")
        for f in object_files:
            if f.endswith(".c"):
                dst = out_dir / (stem + ".c")
                shutil.copy(f, dst)
                print(f"📄  Generated C code saved to {dst}")
                # break
        
        # теперь реальный линковщик
        return cc.create_shared(
            lib_name,
            object_files,
            cc=clang,
            options=clang_opts
        )

    print("🚚  Exporting shared library …")
    exe.export_library(str(so_path), fcompile=my_fcompile)

    # сохраняем мета-данные
    meta = {
        "tvm_version": tvm.__version__,
        "onnx_file": str(onnx_path),
        "input_name": args.input_name,
        "input_shape": args.input_shape,
        "dtype": args.dtype,
        "target": str(target),
        "clang": clang,
        "clang_opts": clang_opts,
        "library": so_path.name
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✅  Build complete.")
    print(" •", so_path)
    print(" •", (out_dir / (stem + ".c")))
    print(f"\n🚀  Готово к копированию на плату:\n  scp {so_path} root@lichee:/opt/models/")

if __name__ == "__main__":
    main()
