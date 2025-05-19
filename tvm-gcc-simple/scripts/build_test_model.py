#!/usr/bin/env python3
# Скрипт для хост-машины: build_test_model.py
import os
import numpy as np
import tvm
from tvm import relax, tir
from tvm.script import tir as T, relax as R

# Создаем модель с использованием TVMScript (рекомендуемый подход в TVM 0.20.0)
@tvm.script.ir_module
class AddOneModule:
    @R.function
    def main(x: R.Tensor((1, 10), "float32")) -> R.Tensor((1, 10), "float32"):
        # Создаем константу 1.0 и прибавляем к каждому элементу входного тензора
        with R.dataflow():
            # Бродкастинг константы 1.0 с формой входа
            one = R.const(1.0, "float32")
            # Выполняем операцию сложения
            y = R.add(x, one)
            # Возвращаем результат
            R.output(y)
        return y

def build_simple_model():
    """Создаёт простую модель: y = x + 1 с использованием TVM 0.20.0"""
    print("Создание и компиляция модели для TVM 0.20.0...")
    
    # Получаем IRModule
    mod = AddOneModule
    
    # Компилируем модель для CPU
    target = tvm.target.Target("llvm", host="llvm")
    
    # Выводим информацию о модуле до компиляции
    print("IR Module до компиляции:")
    print(mod.script())
    
    with tvm.transform.PassContext(opt_level=3):
        # Используем Relax VM API для компиляции модели
        ex_mod = relax.build(mod, target)
    
    # Создаем файлы для запуска на устройстве
    # 1. Сохраняем скомпилированную библиотеку
    ex_mod.export_library("model.so")
    
    # 2. Создаем JSON представление графа
    # В TVM 0.20.0 для VM не используется тот же формат метаданных, что в предыдущих версиях
    # Поэтому создаем минимальный JSON с нужной информацией
    import json
    graph_json = {
        "nodes": [
            {"name": "main", "inputs": [{"name": "x", "shape": [1, 10], "dtype": "float32"}], 
             "outputs": [{"shape": [1, 10], "dtype": "float32"}]}
        ],
        "attrs": {"device_index": 0, "device_type": 1}  # CPU
    }
    
    with open("model.json", "w") as f:
        json.dump(graph_json, f, indent=2)
    
    # 3. Создаем пустой файл параметров (или сохраняем параметры, если они есть)
    with open("model.params", "wb") as f:
        f.write(b'')
    
    print("Модель успешно создана:")
    print(f" - model.so: Библиотека модели ({os.path.getsize('model.so')} байт)")
    print(f" - model.json: Метаданные графа ({os.path.getsize('model.json')} байт)")
    print(f" - model.params: Параметры модели ({os.path.getsize('model.params')} байт)")
    
    # Тестовый запуск на хосте
    print("\nПроверка модели на хосте:")
    dev = tvm.cpu(0)
    vm = relax.VirtualMachine(ex_mod, dev)
    
    # Создаем входные данные
    input_shape = (1, 10)
    dtype = "float32"
    x_data = np.ones(input_shape, dtype=dtype)
    x_nd = tvm.nd.array(x_data, dev)
    
    # Запускаем модель
    result = vm["main"](x_nd)
    
    # Проверяем результаты
    np_result = result.numpy()
    print(f"Вход: {x_data}")
    print(f"Выход: {np_result}")
    print(f"Ожидаемый результат: {x_data + 1.0}")
    print(f"Корректно: {np.allclose(np_result, x_data + 1.0)}")
    
    # Также попробуем перекомпилировать для использования с graph executor 
    print("\nСоздание версии для graph executor...")
    
    # Для graph executor компилируем отдельный модуль
    with tvm.transform.PassContext(opt_level=3):
        try:
            from tvm import relay
            # Попытка использовать relay для создания совместимого модуля
            relay_mod = tvm.relay.Module.from_expr(tvm.relay.var("x", shape=(1, 10), dtype="float32") + tvm.relay.const(1.0))
            lib = relay.build(relay_mod, target)
            # Сохраняем для graph executor
            lib.export_library("graph_model.so")
            with open("graph_model.json", "w") as f:
                f.write(lib.get_graph_json())
            with open("graph_model.params", "wb") as f:
                f.write(lib.get_params())
            print("  Создана версия модели для graph executor!")
        except Exception as e:
            print(f"  Не удалось создать версию для graph executor: {e}")
            print("  Будем использовать VM API для запуска на устройстве")
    
    # Генерируем пример кода для запуска на устройстве RISC-V
    print("\nПример кода для запуска на устройстве RISC-V:")
    print("""
import tvm
from tvm import runtime
import numpy as np

# Загружаем модель
lib = runtime.load_module("model.so")
dev = tvm.runtime.device("cpu", 0)

# Создаем виртуальную машину
vm = tvm.runtime.vm.VirtualMachine(lib, dev)

# Создаем входные данные
x_data = np.ones((1, 10), dtype=np.float32)
x_nd = tvm.nd.array(x_data, dev)

# Запускаем модель
result = vm["main"](x_nd)

# Получаем результат
output = result.numpy()
print(f"Вход: {x_data}")
print(f"Выход: {output}")
print(f"Корректно: {np.allclose(output, x_data + 1.0)}")
    """)

if __name__ == "__main__":
    build_simple_model()
