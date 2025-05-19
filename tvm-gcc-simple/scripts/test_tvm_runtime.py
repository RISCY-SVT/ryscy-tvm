#!/usr/bin/env python3
"""
Минимальный тест для проверки функциональности TVM runtime-only на RISC-V устройстве.
Адаптирован для ограниченного API в runtime-only сборке TVM 0.20.0.
"""

import os
import sys
import time
import ctypes
import unittest
import numpy as np
import json

class TVMRuntimeTest(unittest.TestCase):
    """Тесты для минимальной TVM runtime-only версии на RISC-V устройстве."""
    
    def setUp(self):
        """Настройка окружения для тестов."""
        print("\nИнициализация теста...")
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Определяем расположения важных директорий и файлов
        try:
            import tvm
            
            # Находим библиотеку TVM
            lib_path = None
            if hasattr(tvm._ffi, "_LIB"):
                if hasattr(tvm._ffi._LIB, "_name"):
                    lib_path = tvm._ffi._LIB._name
            
            # Находим директорию пакета TVM
            module_dir = os.path.dirname(os.path.abspath(tvm.__file__))
            self.tvm_dir = module_dir
            self.tvm_lib_path = lib_path
            
            print(f"TVM директория: {self.tvm_dir}")
            print(f"TVM библиотека: {self.tvm_lib_path}")
        except ImportError as e:
            print(f"Ошибка импорта TVM: {e}")
            self.tvm_dir = None
            self.tvm_lib_path = None
    
    def test_import(self):
        """Тест импорта основных модулей TVM."""
        print("\nТест импорта модулей...")
        
        try:
            import tvm
            self.assertEqual(tvm.__version__, "0.20.0", "Версия TVM должна быть 0.20.0")
            self.assertTrue(hasattr(tvm._ffi, "_RUNTIME_ONLY"), "Должен быть флаг _RUNTIME_ONLY")
            self.assertTrue(tvm._ffi._RUNTIME_ONLY, "Флаг _RUNTIME_ONLY должен быть True")
            
            # Проверяем наличие основных модулей
            self.assertTrue(hasattr(tvm, "runtime"), "tvm.runtime должен быть доступен")
            
            # Выводим структуру доступных модулей
            print("\nДоступные модули и подмодули в TVM:")
            self._print_module_structure(tvm)
            
            print("✓ Импорт основных модулей выполнен успешно")
        except ImportError as e:
            self.fail(f"Не удалось импортировать модули TVM: {e}")
    
    def _print_module_structure(self, module, indent=0, prefix="", max_depth=2):
        """Рекурсивно выводит структуру модуля с отступами."""
        if indent > max_depth:
            return
        
        # Получаем имя модуля
        module_name = getattr(module, "__name__", str(module))
        
        # Выводим текущий модуль
        print(f"{' ' * (indent * 2)}{prefix}{module_name}")
        
        # Получаем атрибуты модуля, которые могут быть подмодулями
        if not hasattr(module, "__dict__"):
            return
        
        attributes = dir(module)
        for attr_name in attributes:
            # Пропускаем приватные атрибуты и функции
            if attr_name.startswith("_"):
                continue
            
            try:
                attr = getattr(module, attr_name)
                # Проверяем, является ли атрибут модулем
                if hasattr(attr, "__module__") and hasattr(attr, "__name__"):
                    # Рекурсивно выводим структуру подмодуля
                    self._print_module_structure(attr, indent + 1, f"{attr_name}: ", max_depth)
            except Exception:
                # Пропускаем атрибуты, к которым нет доступа
                pass
    
    def test_runtime_modules(self):
        """Тест наличия и импорта модулей runtime."""
        print("\nТест модулей runtime...")
        
        try:
            import tvm
            from tvm import runtime
            
            # Проверяем наличие класса Module в runtime
            self.assertTrue(hasattr(runtime, "Module"), "runtime.Module должен быть доступен")
            
            # Выводим информацию о доступных атрибутах Module
            print("\nАтрибуты runtime.Module:")
            module_attrs = [attr for attr in dir(runtime.Module) if not attr.startswith("_")]
            print(", ".join(module_attrs))
            
            # Проверяем наличие других классов в runtime
            runtime_attrs = [attr for attr in dir(runtime) if not attr.startswith("_")]
            print("\nДоступные атрибуты в runtime:")
            print(", ".join(runtime_attrs))
            
            # Проверяем наличие contrib модуля
            if hasattr(tvm, "contrib"):
                print("\nДоступные подмодули в tvm.contrib:")
                contrib_modules = [attr for attr in dir(tvm.contrib) if not attr.startswith("_")]
                print(", ".join(contrib_modules))
            
            print("✓ Проверка модулей runtime выполнена успешно")
        except ImportError as e:
            self.fail(f"Не удалось импортировать модули runtime: {e}")
    
    def test_device_creation(self):
        """Тест создания различных устройств."""
        print("\nТест создания устройств...")
        
        try:
            import tvm
            from tvm import runtime
            
            # Создаем CPU устройство
            if hasattr(runtime, "cpu"):
                cpu_dev = runtime.cpu(0)
                self.assertEqual(cpu_dev.device_type, 1, "Тип CPU устройства должен быть 1")
                self.assertEqual(cpu_dev.device_id, 0, "ID CPU устройства должен быть 0")
                print(f"  - Создано устройство {cpu_dev}")
            else:
                print("  ! runtime.cpu не найдено в этой сборке")
            
            # Проверяем создание других устройств
            device_funcs = {}
            for name in ["cpu", "cuda", "gpu", "opencl", "vulkan", "metal"]:
                if hasattr(runtime, name):
                    device_funcs[name] = getattr(runtime, name)
            
            for name, func in device_funcs.items():
                dev = func(0)
                print(f"  - Создано устройство {dev}")
            
            print("✓ Создание устройств выполнено успешно")
        except Exception as e:
            self.fail(f"Ошибка при создании устройств: {e}")
    
    def test_load_library(self):
        """Тест загрузки библиотеки TVM."""
        print("\nТест загрузки нативной библиотеки...")
        
        try:
            import tvm
            
            # Проверяем, что библиотека загружена
            self.assertIsNotNone(tvm._ffi._LIB, "Библиотека TVM должна быть загружена")
            
            # Находим библиотеку
            lib_path = None
            for attr in ["_LIB_NAME", "_LIB"]:
                if hasattr(tvm._ffi, attr):
                    lib_info = getattr(tvm._ffi, attr)
                    if isinstance(lib_info, str):
                        lib_path = lib_info
                    elif hasattr(lib_info, "_name"):
                        lib_path = lib_info._name
                    break
            
            self.assertIsNotNone(lib_path, "Должен быть путь к библиотеке")
            print(f"  - Библиотека загружена из: {lib_path}")
            
            # Пробуем загрузить библиотеку напрямую
            try:
                lib = ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)
                self.assertIsNotNone(lib, "Библиотека должна загружаться напрямую")
                print("  - Библиотека успешно загружена напрямую")
                
                # Проверяем наличие некоторых базовых функций
                print("\n  Проверка наличия базовых функций в библиотеке:")
                key_functions = [
                    "TVMGetLastError",
                    "TVMFuncFree",
                    "TVMFuncCall",
                    "TVMArrayFree",
                    "TVMDeviceCreate"
                ]
                
                for func_name in key_functions:
                    has_func = hasattr(lib, func_name)
                    print(f"  - {func_name}: {'ДОСТУПНО' if has_func else 'ОТСУТСТВУЕТ'}")
                
            except Exception as e:
                print(f"  ! ПРЕДУПРЕЖДЕНИЕ: Не удалось загрузить библиотеку напрямую: {e}")
            
            print("✓ Проверка загрузки библиотеки выполнена успешно")
        except Exception as e:
            self.fail(f"Ошибка при проверке загрузки библиотеки: {e}")
    
    def test_minimal_functionality(self):
        """Тест минимальной функциональности, доступной в runtime-only сборке."""
        print("\nТест минимальной функциональности...")
        
        try:
            import tvm
            from tvm import runtime
            
            # Проверяем основные константы
            self.assertEqual(tvm.__version__, "0.20.0", "Версия должна быть 0.20.0")
            self.assertTrue(tvm._ffi._RUNTIME_ONLY, "Должна быть runtime-only сборка")
            
            # Проверяем доступность класса Module
            self.assertTrue(hasattr(runtime, "Module"), "runtime.Module должен быть доступен")
            
            # Проверяем доступность классов устройств
            for device_type in ["cpu", "cuda", "gpu", "opencl", "vulkan", "metal"]:
                if hasattr(runtime, device_type):
                    print(f"  - Обнаружен тип устройства: {device_type}")
            
            # Собираем информацию о доступных классах и методах
            runtime_classes = {}
            for attr_name in dir(runtime):
                if attr_name.startswith("_"):
                    continue
                
                try:
                    attr = getattr(runtime, attr_name)
                    if isinstance(attr, type) or callable(attr):
                        methods = [method for method in dir(attr) if not method.startswith("_")]
                        runtime_classes[attr_name] = methods
                except Exception:
                    pass
            
            print("\n  Доступные классы и методы в runtime:")
            for class_name, methods in runtime_classes.items():
                if methods:
                    print(f"  - {class_name}: {', '.join(methods[:5])}" + 
                          (f" и еще {len(methods) - 5} методов..." if len(methods) > 5 else ""))
                else:
                    print(f"  - {class_name}")
            
            print("✓ Проверка минимальной функциональности выполнена успешно")
        except Exception as e:
            self.fail(f"Ошибка при проверке минимальной функциональности: {e}")
    
    def test_ctypes_binding(self):
        """Тест прямой работы с C-API через ctypes."""
        print("\nТест прямой работы с C-API через ctypes...")
        
        try:
            import tvm
            
            # Получаем библиотеку
            lib = tvm._ffi._LIB
            
            # Определяем прототипы некоторых функций TVM API
            if hasattr(lib, "TVMGetLastError"):
                get_last_error = ctypes.CFUNCTYPE(ctypes.c_char_p)(
                    ("TVMGetLastError", lib))
                
                # Вызываем функцию
                error_msg = get_last_error()
                print(f"  - TVMGetLastError вернул: {error_msg.decode('utf-8') if error_msg else 'Нет ошибок'}")
            else:
                print("  ! TVMGetLastError отсутствует в библиотеке")
            
            # Проверяем базовое создание устройства
            if hasattr(lib, "TVMDeviceCreate"):
                try:
                    # Определяем структуру DLDevice
                    class DLDevice(ctypes.Structure):
                        _fields_ = [
                            ("device_type", ctypes.c_int),
                            ("device_id", ctypes.c_int)
                        ]
                    
                    # Создаем устройство CPU (тип 1)
                    device = DLDevice(1, 0)  # CPU, device_id = 0
                    print(f"  - Создана структура DLDevice: тип={device.device_type}, id={device.device_id}")
                    
                    # Проверяем создание устройства через C-API
                    device_create = ctypes.CFUNCTYPE(
                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(DLDevice))(
                            ("TVMDeviceCreate", lib))
                    
                    result_device = DLDevice(0, 0)
                    ret = device_create(1, 0, ctypes.byref(result_device))
                    
                    if ret == 0:
                        print(f"  - TVMDeviceCreate успешно создал устройство: тип={result_device.device_type}, id={result_device.device_id}")
                    else:
                        print(f"  ! TVMDeviceCreate вернул код ошибки: {ret}")
                        
                except AttributeError as e:
                    print(f"  ! Ошибка определения функции: {e}")
                except Exception as e:
                    print(f"  ! Ошибка вызова TVMDeviceCreate: {e}")
            else:
                print("  ! TVMDeviceCreate отсутствует в библиотеке")
            
            print("✓ Проверка работы с C-API выполнена")
        except Exception as e:
            self.fail(f"Ошибка при работе с C-API: {e}")
    
    def test_tvm_file_structure(self):
        """Анализ файловой структуры пакета TVM."""
        print("\nАнализ файловой структуры пакета TVM...")
        
        try:
            import tvm
            
            # Находим корневую директорию пакета
            tvm_dir = os.path.dirname(os.path.abspath(tvm.__file__))
            print(f"  Корневая директория TVM: {tvm_dir}")
            
            # Сканируем директорию и выводим информацию о структуре
            print("\n  Структура директорий и файлов:")
            for root, dirs, files in os.walk(tvm_dir, topdown=True):
                # Ограничиваем глубину сканирования
                relative_path = os.path.relpath(root, tvm_dir)
                depth = len(relative_path.split(os.sep)) - 1
                if depth > 2:
                    dirs.clear()  # Не спускаемся глубже
                    continue
                
                # Выводим директорию
                if relative_path != ".":
                    print(f"  {'  ' * (depth)}- Директория: {os.path.basename(root)}")
                
                # Выводим файлы
                for file in sorted(files):
                    if file.endswith(".py") or file.endswith(".so"):
                        print(f"  {'  ' * (depth + 1)}- Файл: {file}")
            
            # Ищем все .so файлы
            print("\n  Найденные библиотеки (.so файлы):")
            for root, _, files in os.walk(tvm_dir):
                for file in files:
                    if file.endswith(".so"):
                        print(f"  - {os.path.join(root, file)}")
            
            print("✓ Анализ файловой структуры выполнен")
        except Exception as e:
            self.fail(f"Ошибка при анализе файловой структуры: {e}")

def run_tests():
    """Запуск всех тестов."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TVMRuntimeTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result

if __name__ == "__main__":
    print("=" * 80)
    print("Минимальные тесты TVM Runtime для RISC-V")
    print("=" * 80)
    
    # Проверяем, что TVM импортируется
    try:
        import tvm
        print(f"TVM версия: {tvm.__version__}")
        print(f"Runtime-only: {tvm._ffi._RUNTIME_ONLY}")
        
        # Получаем информацию о библиотеке
        lib_path = None
        if hasattr(tvm._ffi, "_LIB"):
            if hasattr(tvm._ffi._LIB, "_name"):
                lib_path = tvm._ffi._LIB._name
        
        print(f"Библиотека: {lib_path}")
        print("=" * 80)
    except ImportError as e:
        print(f"ОШИБКА: Не удалось импортировать TVM: {e}")
        sys.exit(1)
    
    # Запускаем тесты
    result = run_tests()
    
    # Выводим сводку
    print("\n" + "=" * 80)
    print("СВОДКА ТЕСТОВ:")
    print(f"Выполнено тестов: {result.testsRun}")
    print(f"Пройдено: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Не пройдено: {len(result.failures)}")
    print(f"Ошибок: {len(result.errors)}")
    
    if len(result.failures) > 0 or len(result.errors) > 0:
        sys.exit(1)
    else:
        print("\nВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        sys.exit(0)
