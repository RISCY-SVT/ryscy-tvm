#!/usr/bin/env bash
set -Eeo pipefail

########################      ПАРАМЕТРЫ     ########################
source ./env.sh
# Проверяем путь к кросс-компилятору
if [ ! -d "$TOOLROOT" ]; then
    echo "Ошибка: Каталог с toolchain не существует: $TOOLROOT"
    exit 1
fi

# Добавляем toolchain в PATH
export PATH=$TOOLROOT/bin:$PATH

# Проверяем доступность компилятора
if ! command -v riscv64-unknown-linux-gnu-gcc &> /dev/null; then
    echo "Ошибка: Компилятор riscv64-unknown-linux-gnu-gcc не найден в PATH"
    exit 1
fi

echo "=================>>> 1. Подготовка окружения"
# Активируем conda окружение (если нужно)
if [ -n "$CONDA_ENV" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# Создаем и очищаем директорию сборки для RISC-V
echo "=================>>> 2. Конфигурация CMake"
[ -d "$Dev_BUILD" ] && rm -rf "${Dev_BUILD:?}"
mkdir -p "$Dev_BUILD"
cp "$TVM_HOME"/../config.cmake.lichee "$Dev_BUILD/config.cmake"

# Переход в директорию сборки
cd "$Dev_BUILD" || exit 1

echo "=================>>> 3. Сборка RISC-V runtime"
# Вызываем CMake с правильными флагами для кросс-компиляции
cmake .. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_C_COMPILER=$TOOLROOT/bin/riscv64-unknown-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=$TOOLROOT/bin/riscv64-unknown-linux-gnu-g++ \
    -DCMAKE_C_FLAGS="$RISCV_CFLAGS" \
    -DCMAKE_CXX_FLAGS="$RISCV_CFLAGS" \
    -DCMAKE_FIND_ROOT_PATH=$TOOLROOT/riscv64-unknown-linux-gnu \
    -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
    -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DUSE_ALTERNATIVE_LINKER=OFF

# Собираем только runtime
make -j$JOBS runtime

# Создание информации о версии
echo "=================>>> 4. Проверка собранной библиотеки"
file "$Dev_BUILD/libtvm_runtime.so"
readelf -A "$Dev_BUILD/libtvm_runtime.so" | grep -E 'Tag_RISCV|ABI|Arch'

echo "=================>>> 5. Создание минимального пакета TVM runtime для RISC-V"
# Создаем временную директорию для сборки пакета
WHEEL_BUILD_DIR="$TVM_HOME/python_riscv"
rm -rf "$WHEEL_BUILD_DIR" || true
mkdir -p "$WHEEL_BUILD_DIR/tvm"

# Вместо копирования всего Python пакета, мы создадим минимальную структуру
# для runtime-only сборки
mkdir -p "$WHEEL_BUILD_DIR/tvm/_ffi"
mkdir -p "$WHEEL_BUILD_DIR/tvm/runtime"

# Копируем библиотеки
cp "$Dev_BUILD/libtvm_runtime.so" "$WHEEL_BUILD_DIR/tvm/"

# Создаем пустой файл libtvm_pkg_root, чтобы TVM мог найти корень пакета
touch "$WHEEL_BUILD_DIR/tvm/libtvm_pkg_root"

# Создаем базовые файлы инициализации
cat > "$WHEEL_BUILD_DIR/tvm/__init__.py" << 'EOF'
"""TVM: An Efficient Tensor IR Stack for Deep Learning Systems"""
import os
import sys
import numpy as np

# Import minimal modules for runtime
from ._ffi.base import TVMError, __version__, _RUNTIME_ONLY
import tvm.runtime
from tvm.runtime import *

__all__ = ["runtime", "__version__"]
EOF

cat > "$WHEEL_BUILD_DIR/tvm/_ffi/__init__.py" << 'EOF'
"""FFI: Foreign Function Interface for TVM."""
import sys
import os
import ctypes
import types
import numpy as np
import warnings

from .base import TVMError, register_error, _LIB_NAME
from .base import _LIB, _RUNTIME_ONLY, check_call
from .base import py_str, c_str, string_types, get_last_ffi_error
from .registry import register_object, register_func, register_extension
EOF

cat > "$WHEEL_BUILD_DIR/tvm/runtime/__init__.py" << 'EOF'
"""Package tvm.runtime"""
from .packed_func import PackedFunc
from .module import Module
from .ndarray import NDArray, device, cpu, cuda, gpu, opencl, cl, vulkan, metal, mtl
from .container import String
from .ndarray import array
from .object_generic import ObjectGeneric, convert_to_object
from ..contrib import graph_executor
EOF

# Создаем runtime модули
cat > "$WHEEL_BUILD_DIR/tvm/runtime/module.py" << 'EOF'
"""Module container of TVM."""
import os
import ctypes
import numpy as np
from .._ffi.base import _LIB, check_call, c_str, string_types, _RUNTIME_ONLY
from .._ffi.base import py_str, _FFI_MODE
from .._ffi.registry import register_object
from .ndarray import NDArray, context, empty
from .packed_func import PackedFunc

class Module(object):
    """Runtime Module."""
    def __init__(self, handle):
        self.handle = handle
        self._entry = None
        self._imports = None
        self._type_key = None
        self._format = None
        
    def __del__(self):
        if _LIB:
            # In runtime-only mode we need to be careful about cleanup
            try:
                free_module = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(
                    ('TVMModFree', _LIB))
                if free_module:
                    check_call(free_module(self.handle))
            except:
                pass

    def __call__(self, *args):
        if self._entry is None:
            if _RUNTIME_ONLY:
                raise RuntimeError("Cannot call module in runtime-only mode")
            self._entry = self.get_function("__tvm_main__")
        return self._entry(*args)

    def get_function(self, name, query_imports=False):
        """Get function from the module."""
        if not query_imports and self._imports:
            for module in self._imports:
                try:
                    return module.get_function(name)
                except TVMError:
                    pass
                
        func_handle = ctypes.c_void_p()
        get_func = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p))(
                ('TVMModGetFunction', _LIB))
                
        check_call(get_func(self.handle, c_str(name), ctypes.byref(func_handle)))
                
        if not func_handle.value:
            raise AttributeError(f"Module has no function {name}")
        return PackedFunc(func_handle, False)

    @property
    def type_key(self):
        """Get type key of the module."""
        if self._type_key is None:
            func = self.get_function("_type_key", True)
            self._type_key = func()
        return self._type_key

    @property
    def format(self):
        """Get format of the module."""
        if self._format is None:
            func = self.get_function("_format", True)
            self._format = func()
        return self._format

    @staticmethod
    def load(path, fmt=""):
        """Load module from file."""
        # Use TVMModuleLoad to load the module
        if not os.path.exists(path):
            raise ValueError(f"Module file {path} does not exist")
        
        load_mod = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p))(
                ('TVMModuleLoad', _LIB))
        
        mod_handle = ctypes.c_void_p()
        check_call(load_mod(c_str(path), c_str(fmt), ctypes.byref(mod_handle)))
        return Module(mod_handle)

    @staticmethod
    def enabled(target):
        """Whether module is enabled for target."""
        # Runtime-only mode can only check for a few targets
        enabled_targets = {"cpu", "cuda", "opencl", "metal", "vulkan"}
        return target in enabled_targets
EOF

cat > "$WHEEL_BUILD_DIR/tvm/runtime/packed_func.py" << 'EOF'
"""PackedFunc object used in TVM runtime."""
import ctypes
import numpy as np
from .._ffi.base import _LIB, check_call, c_str, py_str, _FFI_MODE, _RUNTIME_ONLY

class PackedFunc(object):
    """PackedFunc object."""
    
    __slots__ = ["handle", "is_global"]
    
    def __init__(self, handle, is_global):
        """Initialize the function with handle."""
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global and _LIB:
            free_fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
                ('TVMPackedFuncFree', _LIB))
            free_fn(self.handle)

    def __call__(self, *args):
        """Call the function with arguments."""
        # In runtime-only mode, we implement a simplified version
        # that only supports basic types
        temp_args = []
        values = []
        type_codes = []
        
        for arg in args:
            if isinstance(arg, (int, float, str, bool)):
                values.append(arg)
                if isinstance(arg, int):
                    type_codes.append(0)  # INT
                elif isinstance(arg, float):
                    type_codes.append(2)  # FLOAT
                elif isinstance(arg, str):
                    type_codes.append(10)  # STR
                elif isinstance(arg, bool):
                    type_codes.append(6)  # BOOL
            elif arg is None:
                values.append(None)
                type_codes.append(4)  # NULL
            elif isinstance(arg, np.ndarray):
                values.append(arg)
                type_codes.append(7)  # NDARRAY
            elif isinstance(arg, ctypes.c_void_p):
                values.append(arg)
                type_codes.append(3)  # HANDLE
            else:
                raise TypeError(f"Unsupported type: {type(arg)}")
        
        # Call the function (simplified implementation)
        ret_val = ctypes.c_void_p()
        ret_type_code = ctypes.c_int()
        
        # Not implemented in runtime-only mode
        raise NotImplementedError(
            "PackedFunc.__call__ is not fully implemented in runtime-only mode")
EOF

cat > "$WHEEL_BUILD_DIR/tvm/runtime/ndarray.py" << 'EOF'
"""Runtime NDArray API"""
import ctypes
import numpy as np
from .._ffi.base import _LIB, check_call, c_str, _FFI_MODE, _RUNTIME_ONLY

# Define device types
class Device(object):
    """Device context for NDArray."""
    
    def __init__(self, device_type, device_id=0):
        self.device_type = device_type
        self.device_id = device_id

    def __eq__(self, other):
        return (self.device_type == other.device_type and 
                self.device_id == other.device_id)

    def __str__(self):
        if self.device_type == 1:  # CPU
            return "cpu(%d)" % self.device_id
        elif self.device_type == 2:  # GPU/CUDA
            return "cuda(%d)" % self.device_id
        elif self.device_type == 4:  # OpenCL
            return "opencl(%d)" % self.device_id
        elif self.device_type == 7:  # Vulkan
            return "vulkan(%d)" % self.device_id
        elif self.device_type == 8:  # Metal
            return "metal(%d)" % self.device_id
        else:
            return "device(%d, %d)" % (self.device_type, self.device_id)

# Device API
def device(device_type, device_id=0):
    """Construct a device."""
    return Device(device_type, device_id)

def cpu(device_id=0):
    """Construct a CPU device."""
    return Device(1, device_id)

def cuda(device_id=0):
    """Construct a CUDA device."""
    return Device(2, device_id)

def gpu(device_id=0):
    """Construct a GPU device."""
    return cuda(device_id)

def opencl(device_id=0):
    """Construct a OpenCL device."""
    return Device(4, device_id)

def cl(device_id=0):
    """Construct a OpenCL device."""
    return opencl(device_id)

def vulkan(device_id=0):
    """Construct a Vulkan device."""
    return Device(7, device_id)

def metal(device_id=0):
    """Construct a metal device."""
    return Device(8, device_id)

def mtl(device_id=0):
    """Construct a metal device."""
    return metal(device_id)

class NDArray(object):
    """NDArray in TVM."""
    
    def __init__(self, handle):
        """Initialize the function with handle."""
        self.handle = handle
        self._shape = None
        self._dtype = None

    def __del__(self):
        if _LIB:
            try:
                free_fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
                    ('TVMArrayFree', _LIB))
                free_fn(self.handle)
            except:
                pass

    @property
    def shape(self):
        """Shape of this array."""
        # Not fully implemented in runtime-only
        return (1,)

    @property
    def dtype(self):
        """Type of this array."""
        # Not fully implemented in runtime-only
        return "float32"

    def numpy(self):
        """Convert to numpy array."""
        # Not fully implemented in runtime-only
        raise NotImplementedError("NDArray.numpy() is not implemented in runtime-only mode")
        
def array(arr, ctx=cpu(0)):
    """Create an array from source arr."""
    # Not fully implemented in runtime-only
    raise NotImplementedError("array() is not implemented in runtime-only mode")

def empty(shape, dtype="float32", ctx=cpu(0)):
    """Create an empty array."""
    # Not fully implemented in runtime-only
    raise NotImplementedError("empty() is not implemented in runtime-only mode")

# Context is a synonym of Device
context = device
EOF

cat > "$WHEEL_BUILD_DIR/tvm/runtime/container.py" << 'EOF'
"""Container types used in TVM."""
import ctypes
from .._ffi.base import string_types

class String(object):
    """TVM String class."""
    
    def __init__(self, value):
        """Initialize with python string."""
        self.value = value

    def __str__(self):
        return self.value

    def __repr__(self):
        return repr(self.value)
EOF

cat > "$WHEEL_BUILD_DIR/tvm/runtime/object_generic.py" << 'EOF'
"""Common implementation of object generic methods."""

class ObjectGeneric(object):
    """Base class for all classes that can be converted to object."""
    
    def __init__(self):
        pass

    def asobj(self):
        """Convert to object."""
        raise NotImplementedError()

def convert_to_object(value):
    """Convert a Python value to corresponding object."""
    if isinstance(value, ObjectGeneric):
        return value.asobj()
    return value
EOF

# Создаем graph_executor модуль
mkdir -p "$WHEEL_BUILD_DIR/tvm/contrib"
cat > "$WHEEL_BUILD_DIR/tvm/contrib/__init__.py" << 'EOF'
"""Contrib modules"""
EOF

cat > "$WHEEL_BUILD_DIR/tvm/contrib/graph_executor.py" << 'EOF'
"""Graph executor that executes graph containing TVM PackedFunc."""
import numpy as np
import json
from ..runtime import Module
from ..runtime.container import String
from ..runtime.object_generic import convert_to_object

def create(graph_json, libmod, ctx):
    """Create a runtime executor module given a graph and module."""
    if not isinstance(graph_json, (str, bytes, bytearray)):
        try:
            graph_json = json.dumps(graph_json)
        except TypeError:
            raise ValueError("Type of graph_json must be string or dict")
    
    return GraphModule(libmod["runtime.GraphExecutor"](graph_json, libmod, *ctx))

def load_module(module_file, fmt=""):
    """Load module from file."""
    return Module.load(module_file, fmt)

class GraphModule(object):
    """Graph executor module."""
    
    def __init__(self, module):
        self.module = module
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._get_num_outputs = module["get_num_outputs"]
        
    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module."""
        if key is not None:
            self._set_input(key, value)
        for k, v in params.items():
            self._set_input(k, v)

    def run(self):
        """Run forward execution of the graph."""
        self._run()

    def get_input(self, index, out=None):
        """Get input at index."""
        if out:
            self._get_input(index, out)
            return out
        return self._get_input(index)

    def get_output(self, index, out=None):
        """Get output at index."""
        if out:
            self._get_output(index, out)
            return out
        return self._get_output(index)

    def get_num_outputs(self):
        """Get number of outputs."""
        return self._get_num_outputs()

    def load_params(self, params_bytes):
        """Load parameters from serialized byte array."""
        self.module["load_params"](params_bytes)
EOF

# Создаем registry модуль
cat > "$WHEEL_BUILD_DIR/tvm/_ffi/registry.py" << 'EOF'
"""Registry for TVM functions."""
import ctypes
from .base import _LIB, check_call, py_str, c_str, _RUNTIME_ONLY

def register_object(type_key=None):
    """Decorator for registering object."""
    def wrap(cls):
        return cls
    return wrap

def register_func(func_name, f=None, override=False):
    """Register global function."""
    # In runtime-only mode, we don't need this functionality
    return f

def register_extension(cls, fcreate=None):
    """Register extension class to TVM."""
    # In runtime-only mode, we don't need this functionality
    return cls
EOF

# Создаем более полную версию base.py
cat > "$WHEEL_BUILD_DIR/tvm/_ffi/base.py" << 'EOF'
"""Base definitions for FFI."""
import ctypes
import sys
import os
import numpy as np
from . import libinfo

# версия
__version__ = "0.20.0"

# Marker for runtime only
_RUNTIME_ONLY = True

# FFI mode
_FFI_MODE = "ctypes"

# типы строк: в Python 3 unicode и str это одно и то же
string_types = (str,)

class TVMError(Exception):
    """Default error thrown by TVM functions."""
    pass

def _register_error():
    """Register the error classes."""
    def __init__(self, msg):
        super(TVMError, self).__init__(msg)
        self.message = msg

    TVMError.__init__ = __init__
    return TVMError

TVMError = _register_error()

def register_error(func_name=None, cls=None):
    """Регистрация ошибок."""
    if cls is None:
        cls = TVMError
    return cls

def _load_lib():
    """Load library by searching possible path."""
    lib_path = libinfo._get_paths()["runtime"]
    lib = ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)
    return lib, os.path.basename(lib_path)

# Library instance
_LIB, _LIB_NAME = _load_lib()

def check_call(ret):
    """Check the return value of C API call.

    This function will raise exception when error occurs.
    Wrap every API call with this function.

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        raise TVMError(py_str(get_last_ffi_error()))

def c_str(string):
    """Create ctypes char * from a Python string.

    Parameters
    ----------
    string : string type
        Python string

    Returns
    -------
    str : c_char_p
        A char pointer that can be passed to C API
    """
    return ctypes.c_char_p(string.encode("utf-8"))

def py_str(c_str):
    """Convert C string to Python string."""
    return c_str.decode("utf-8")

def get_last_ffi_error():
    """Get the last error from the FFI."""
    if not hasattr(_LIB, "TVMGetLastError"):
        return "unknown error"
    
    get_last_error = ctypes.CFUNCTYPE(ctypes.c_char_p)(
        ("TVMGetLastError", _LIB))
    
    if get_last_error:
        return get_last_error()
    return "unknown error"
EOF

# Создаем libinfo.py для поиска библиотек
cat > "$WHEEL_BUILD_DIR/tvm/_ffi/libinfo.py" << 'EOF'
"""Library information."""
import os
import sys
import ctypes

def find_lib_path(name=None, search_path=None, optional=False):
    """Find dynamic library files.

    Parameters
    ----------
    name : list of str
        List of names to be found.

    Returns
    -------
    lib_path : str
        Path to the dynamic library.
    """
    use_runtime = True
    
    # This is a runtime-only build for RISC-V 
    if not name:
        if use_runtime:
            name = ["libtvm_runtime.so"]
        else:
            name = ["libtvm.so", "libtvm_runtime.so"]
    
    # Ищем библиотеку в каталоге пакета tvm
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_search_paths = [os.path.join(curr_path, "..", p) for p in ["", ".."]]
    
    lib_path = None
    for path in lib_search_paths:
        path = os.path.abspath(path)  # normalized path
        for libname in name:
            full_path = os.path.join(path, libname)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                lib_path = full_path
                break
        if lib_path:
            break
    
    if not lib_path and not optional:
        message = f"Cannot find libraries: {name}\n"
        message += "List of candidates:\n"
        for libname in name:
            for path in lib_search_paths:
                message += os.path.abspath(os.path.join(path, libname)) + "\n"
        raise RuntimeError(message)
    
    return lib_path

def _get_paths():
    """Get the paths for various components in the TVM package"""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    root_path = os.path.abspath(os.path.join(curr_path, ".."))
    lib_path = os.path.join(root_path, "libtvm_runtime.so")
    return {
        "root": root_path,
        "runtime": lib_path,
        "lib": lib_path,
    }
EOF

# Создаем setup.py для сборки wheel
cat > "$WHEEL_BUILD_DIR/setup.py" << EOF
import os
import setuptools
from setuptools import find_packages

def get_version():
    return "0.20.0"

setuptools.setup(
    name="tvm",
    version=get_version(),
    license="Apache-2.0",
    description="TVM Runtime for RISC-V (LicheePi4A)",
    author="Apache TVM",
    author_email="dev@tvm.apache.org",
    url="https://github.com/apache/tvm",
    packages=find_packages(),
    package_data={
        'tvm': ['*.so', 'libtvm_pkg_root'],
    },
    install_requires=[
        "numpy",
        "packaging"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
)
EOF

# Создаем файл setup.cfg для установки платформы
cat > "$WHEEL_BUILD_DIR/setup.cfg" << EOF
[bdist_wheel]
plat-name = any
universal = 1
EOF

# Переходим в директорию сборки wheel и собираем пакет
cd "$WHEEL_BUILD_DIR"

# Генерируем универсальное колесо
python setup.py bdist_wheel

# Переносим wheel в общую директорию для дистрибутивов
mkdir -p "$TVM_HOME/dist/riscv64"
cp dist/*.whl "$TVM_HOME/dist/riscv64/"

echo
echo "=================>>> Готово!"
echo "TVM RISC-V Runtime wheel находится в $TVM_HOME/dist/riscv64/"
echo "Для установки на устройстве LicheePi4A используйте:"
echo "pip install $TVM_HOME/dist/riscv64/tvm-0.20.0-py2.py3-none-any.whl"
echo
