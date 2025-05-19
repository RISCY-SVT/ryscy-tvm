#!/usr/bin/env python3
import tvm
import sys

def explore_module(module, prefix="", max_depth=3, current_depth=0):
    if current_depth > max_depth:
        return
    
    for name in dir(module):
        if name.startswith("_"):
            continue
        
        try:
            attr = getattr(module, name)
            full_name = f"{prefix}.{name}" if prefix else name
            
            if "onnx" in name.lower():
                print(f"Found potential ONNX module: {full_name}")
            
            # Recursively explore submodules
            if hasattr(attr, "__module__") and attr.__module__.startswith("tvm"):
                print(f"Exploring submodule: {full_name}")
                explore_module(attr, full_name, max_depth, current_depth + 1)
        except Exception as e:
            print(f"Error exploring {name}: {e}")

print(f"TVM version: {tvm.__version__}")
print("Exploring TVM module structure to find ONNX-related components...")
explore_module(tvm)
