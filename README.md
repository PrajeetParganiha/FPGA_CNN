# YOLOv5 FPGA Deployment with FINN on Pynq-Z2

Quantized YOLOv5 object detection deployed on a Pynq-Z2 FPGA using the FINN framework. The model is trained with Brevitas for quantization-aware training (QAT), exported to ONNX, and compiled through the FINN dataflow pipeline for FPGA inference.

---

## Overview

This project implements a quantized YOLOv5 model trained on a VOC dataset and targets the Pynq-Z2 FPGA board (Xilinx XC7Z020). The full pipeline covers model training, quantization, FINN compilation, RTL simulation, and hardware deployment.

**Key results from FINN build:**
- RTL simulation: 346,112 inputs consumed, 25,350 outputs produced
- Clock cycles: 22,053,859
- Target throughput: 10 FPS at 100 MHz

---

## Pipeline

```
YOLOv5 Training (Brevitas QAT)
        ↓
ONNX Export (best.finn.onnx)
        ↓
FINN Build Pipeline (19 steps)
        ↓
RTL Simulation & Verification
        ↓
Vivado Synthesis (Pynq-Z2)
        ↓
Deployment & Inference
```

---

## Requirements

- [FINN](https://github.com/Xilinx/finn) (Docker-based)
- Vivado 2022.1
- Pynq-Z2 board with image v2.7
- Python 3.8 (on Pynq board)
- PyTorch 1.8.1 (armv7l)

---

## Installation

### Host (FINN Docker)

Clone this repo and start the FINN Docker container:

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

Follow the [FINN installation guide](https://finn.readthedocs.io/en/latest/getting_started.html) to set up the Docker environment.

### Pynq-Z2 Board

Flash the Pynq-Z2 with the [v2.7 image](https://www.pynq.io/board.html). Then install dependencies:

```bash
pip install torch-1.8.1-cp38-cp38-linux_armv7l.whl
pip install torchvision-0.9.1-cp38-cp38-linux_armv7l.whl
pip install bitstream
```

PyTorch armv7 wheels can be downloaded from [KumaTea/pytorch-arm](https://github.com/KumaTea/pytorch-arm/releases/tag/v1.8.1).

---

## Usage

### 1. Build the FINN Dataflow Accelerator

Open `build.ipynb` in the FINN Jupyter environment and run all cells. The build config targets the Pynq-Z2 at 10 FPS:

```python
cfg = build_cfg.DataflowBuildConfig(
    output_dir="output_yolov5_pynq",
    target_fps=10,
    synth_clk_period_ns=10.0,
    board="Pynq-Z2",
    shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
    ],
)
```

### 2. Copy Deployment Package to Pynq-Z2

```bash
scp -r output_yolov5_pynq/deploy/ xilinx@<PYNQ_IP>:/home/xilinx/finn_deploy/
```

### 3. Run Inference on the Board

Open `inference.ipynb` in the Pynq Jupyter environment (navigate to `http://<PYNQ_IP>:9090`) and run inference on images or video.

---

## FINN Build Steps

The full 19-step FINN pipeline that was executed:

| Step | Description |
|------|-------------|
| step_qonnx_to_finn | Convert QONNX to FINN IR |
| step_tidy_up | Clean up the model graph |
| step_streamline | Streamline operations |
| step_convert_to_hw | Convert to HW layers |
| step_create_dataflow_partition | Partition the dataflow graph |
| step_specialize_layers | Specialize layer implementations |
| step_target_fps_parallelization | Set parallelization for target FPS |
| step_apply_folding_config | Apply folding configuration |
| step_minimize_bit_width | Minimize bit widths |
| step_generate_estimate_reports | Generate resource/performance estimates |
| step_hw_codegen | Generate HW code (HLS/RTL) |
| step_hw_ipgen | Run HLS synthesis |
| step_set_fifo_depths | Set FIFO depths via RTL simulation |
| step_create_stitched_ip | Stitch IP blocks in Vivado |
| step_measure_rtlsim_performance | Measure RTL simulation performance |
| step_out_of_context_synthesis | Out-of-context synthesis |
| step_synthesize_bitfile | Full Vivado synthesis + P&R |
| step_make_pynq_driver | Generate Pynq driver |
| step_deployment_package | Package for deployment |

---

## Known Issues

### FIFO Depth Error in Vivado

During `step_create_stitched_ip`, Vivado may throw:
```
ERROR: Value '524288' is out of the range for parameter 'FIFO depth(FIFO_DEPTH)'
Valid values are: 16, 32, 64, ..., 32768
```

**Fix:** Patch the FINN source to cap FIFO depth at 32768:

```python
# In finn/src/finn/custom_op/fpgadataflow/rtl/streamingfifo_rtl.py
# Change line ~225 from:
"[get_bd_cells /%s/fifo]" % (depth, node_name)
# To:
"[get_bd_cells /%s/fifo]" % (min(depth, 32768), node_name)
```

Then rerun from `step_create_stitched_ip`.

---

## Project Structure

```
.
├── best.finn.onnx                  # Quantized YOLOv5 ONNX model
├── build.ipynb                     # FINN build notebook
├── output_yolov5_pynq/
│   ├── report/                     # Estimate reports
│   │   ├── estimate_network_performance.json
│   │   ├── estimate_layer_resources.json
│   │   └── estimate_layer_cycles.json
│   ├── intermediate_models/        # Checkpoints for each build step
│   └── deploy/                     # Final deployment package
│       ├── finn-accel.bit
│       ├── finn-accel.hwh
│       └── driver.py
└── README.md
```

---

## References

- [FINN Framework](https://github.com/Xilinx/finn)
- [Brevitas QAT Library](https://github.com/Xilinx/brevitas)
- [quantized-yolov5 Training](https://github.com/sefaburakokcu/quantized-yolov5)
- [LPYOLO Paper](https://arxiv.org/abs/2207.10482)
- [End-to-End Deployment Guide](https://medium.com/@bestamigunay1/end-to-end-deployment-of-lpyolo-low-precision-yolo-for-face-detection-on-fpga-13c3284ed14b)
- [Pynq-Z2 Board](http://www.pynq.io/board.html)
