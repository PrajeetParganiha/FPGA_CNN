import os
os.environ["FINN_BUILD_DIR"] = "/tmp/finn_dev_" + os.environ.get("USER", "user")
os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)
print("FINN build dir:", os.environ["FINN_BUILD_DIR"])
import logging
import sys
import warnings
from qonnx.core.modelwrapper import ModelWrapper
from finn.builder.build_dataflow import build_dataflow_cfg
import finn.builder.build_dataflow_config as build_cfg

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout
)

# Filename
model_file = 'best.finn.onnx'
output_dir = "output_yolov5_pynq"

# Validation Check
if not os.path.exists(model_file):
    print(f"❌ ERROR: {model_file} not found!")
    print(f"Current Directory: {os.getcwd()}")
    print(f"Files here: {os.listdir('.')}")
else:
    # Configure the Build
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=output_dir,
        target_fps=10,
        synth_clk_period_ns=10.0,
        board="Pynq-Z2",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,  # <-- COMMENT THIS OUT
            build_cfg.DataflowOutputType.PYNQ_DRIVER,  # <-- AND THIS
        ],
        verbose=True,
        save_intermediate_models=True
    )
    
    print("🚀 Starting FINN Build for VOC10 model...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_dataflow_cfg(model_file, cfg)
    
    print("✅ Build complete!")