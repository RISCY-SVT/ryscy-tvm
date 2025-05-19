#!/usr/bin/env python3
# compile_yolov5n_for_licheepi4a.py
# Complete script to compile YOLOv5n for LicheePi4A using TVM 0.20.0 with Relax API

import os
import sys
import time
import numpy as np
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend import from_onnx
from tvm.relax.transform import LegalizeOps, DecomposeOps, AnnotateTIROpPattern, FuseOps
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("YOLOv5n-Compiler")

def setup_directories():
    """Setup required directories for model compilation"""
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "../data"))
    
    # Create necessary directories
    models_dir = os.path.join(data_dir, "models")
    output_dir = os.path.join(models_dir, "tvm_compiled")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    return data_dir, models_dir, output_dir

def load_onnx_model(onnx_path):
    """Load ONNX model from path"""
    logger.info(f"Loading ONNX model from {onnx_path}")
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX model not found at {onnx_path}")
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
    
    try:
        model = onnx.load(onnx_path)
        logger.info("ONNX model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {str(e)}")
        raise

def convert_to_relax(onnx_model, input_shape):
    """Convert ONNX model to TVM Relax IRModule"""
    logger.info(f"Converting ONNX model to Relax IRModule with input shape: {input_shape}")
    try:
        # Use from_onnx to convert the model to a Relax IRModule
        mod = from_onnx(onnx_model, shape_dict={"images": input_shape})
        logger.info("Model converted to Relax IRModule successfully")
        return mod
    except Exception as e:
        logger.error(f"Failed to convert model to Relax: {str(e)}")
        raise

def optimize_relax_module(mod):
    """Apply optimization passes to the Relax IRModule"""
    logger.info("Applying optimization passes to the Relax module")
    try:
        # Define a sequence of optimization passes
        seq = tvm.transform.Sequential([
            # Legalize operations to ensure all ops can be properly handled
            LegalizeOps(),
            # Decompose complex ops into simpler ones
            DecomposeOps(),
            # Annotate TIR op patterns for better fusion
            AnnotateTIROpPattern(),
            # Fuse operations together for better performance
            FuseOps(fuse_opt_level=2)
        ])
        
        # Apply the optimization passes
        optimized_mod = seq(mod)
        logger.info("Optimization passes applied successfully")
        
        # Print the optimized module structure for debugging
        logger.debug("Optimized IRModule structure:")
        logger.debug(optimized_mod.script())
        
        return optimized_mod
    except Exception as e:
        logger.error(f"Failed to optimize Relax module: {str(e)}")
        raise

def configure_target_for_riscv():
    """Configure target platform for RISC-V (LicheePi4A with T-Head C910)"""
    logger.info("Configuring target platform for RISC-V (LicheePi4A)")
    try:
        # Define the target architecture for LicheePi4A
        # T-Head C910 supports RV64GCV (G=base+MulDiv+Atomics+FP+D), C=Compressed, V=Vector extensions
        target = tvm.target.Target(
            "llvm -mtriple=riscv64-unknown-linux-gnu -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c",
            host="llvm -mtriple=riscv64-unknown-linux-gnu"
        )
        
        logger.info(f"Target configured: {target}")
        logger.info(f"Target features: {target.attrs}")
        return target
    except Exception as e:
        logger.error(f"Failed to configure target: {str(e)}")
        raise

def build_relax_module(optimized_mod, target):
    """Build the optimized Relax module for the target platform"""
    logger.info("Building the optimized Relax module")
    try:
        # Use PassContext with opt_level=3 for maximum optimization
        with tvm.transform.PassContext(opt_level=3):
            # Build the Relax module for the target
            exe_mod = relax.build(optimized_mod, target=target)
            
        logger.info("Relax module built successfully")
        return exe_mod
    except Exception as e:
        logger.error(f"Failed to build Relax module: {str(e)}")
        raise

def save_compiled_model(exe_mod, output_dir, model_name="yolov5n_riscv"):
    """Save the compiled model to the output directory"""
    logger.info(f"Saving compiled model to {output_dir}")
    try:
        # Define output file paths
        lib_path = os.path.join(output_dir, f"{model_name}.so")
        json_path = os.path.join(output_dir, f"{model_name}.json")
        params_path = os.path.join(output_dir, f"{model_name}.params")
        
        # Export the compiled library
        exe_mod.export_library(lib_path)
        logger.info(f"Compiled library saved to {lib_path}")
        
        # Get and save the graph JSON
        # Access metadata through the correct method chain
        metadata = exe_mod.get_executor_codegen_metadata()
        graph_json = metadata.executor_factory.get_graph_json()
        
        with open(json_path, "w") as f:
            f.write(graph_json)
        logger.info(f"Graph JSON saved to {json_path}")
        
        # Save parameters
        with open(params_path, "wb") as f:
            f.write(exe_mod.get_params())
        logger.info(f"Parameters saved to {params_path}")
        
        # Print file sizes for verification
        lib_size = os.path.getsize(lib_path) / (1024 * 1024)  # Size in MB
        params_size = os.path.getsize(params_path) / (1024 * 1024)  # Size in MB
        logger.info(f"Library size: {lib_size:.2f} MB")
        logger.info(f"Parameters size: {params_size:.2f} MB")
        
        return lib_path, json_path, params_path
    except Exception as e:
        logger.error(f"Failed to save compiled model: {str(e)}")
        raise

def verify_build_correctness(lib_path, json_path, params_path):
    """Verify that the built files exist and have non-zero size"""
    logger.info("Verifying build correctness")
    try:
        # Check if files exist
        files_exist = all(os.path.exists(f) for f in [lib_path, json_path, params_path])
        
        # Check if files have content
        files_have_content = all(os.path.getsize(f) > 0 for f in [lib_path, json_path, params_path])
        
        if files_exist and files_have_content:
            logger.info("Build verification successful: All files exist and have content")
            return True
        else:
            missing_files = [f for f in [lib_path, json_path, params_path] if not os.path.exists(f)]
            empty_files = [f for f in [lib_path, json_path, params_path] if os.path.exists(f) and os.path.getsize(f) == 0]
            
            if missing_files:
                logger.error(f"Build verification failed: Missing files: {missing_files}")
            if empty_files:
                logger.error(f"Build verification failed: Empty files: {empty_files}")
            
            return False
    except Exception as e:
        logger.error(f"Failed to verify build: {str(e)}")
        raise


def main():
    """Main function to compile YOLOv5n for LicheePi4A"""
    logger.info("Starting YOLOv5n compilation for LicheePi4A")
    start_time = time.time()
    
    try:
        # 1. Setup directories
        data_dir, models_dir, output_dir = setup_directories()
        
        # 2. Load ONNX model
        onnx_path = os.path.join(models_dir, "yolov5n.onnx")
        onnx_model = load_onnx_model(onnx_path)
        
        # 3. Convert ONNX to Relax IRModule
        input_shape = (1, 3, 640, 640)  # [batch_size, channels, height, width]
        mod = convert_to_relax(onnx_model, input_shape)
        
        # 4. Optimize Relax module
        optimized_mod = optimize_relax_module(mod)
        
        # 5. Configure target for RISC-V
        target = configure_target_for_riscv()
        
        # 6. Build Relax module
        exe_mod = build_relax_module(optimized_mod, target)
        
        # 7. Save compiled model
        lib_path, json_path, params_path = save_compiled_model(exe_mod, output_dir)
        
        # 8. Verify build correctness
        verify_build_correctness(lib_path, json_path, params_path)
        
        # 9. Generate test script
        test_script_path = generate_test_script(output_dir)
        
        # 10. Print completion message
        elapsed_time = time.time() - start_time
        logger.info(f"Compilation completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Compiled model files saved to {output_dir}")
        logger.info(f"Test script saved to {test_script_path}")
        logger.info("To run on LicheePi4A, transfer these files and execute the test script")
        
    except Exception as e:
        logger.error(f"Compilation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())