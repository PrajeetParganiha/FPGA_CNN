# YOLOv5 FPGA Deployment with FINN on Pynq-Z2

Quantized YOLOv5 object detection on a Pynq-Z2 FPGA using the FINN framework. The model was trained using [quantized-yolov5](https://github.com/sefaburakokcu/quantized-yolov5) with Brevitas for quantization-aware training (QAT), exported to ONNX, and compiled through the FINN dataflow pipeline targeting the Pynq-Z2 board.

This project follows the deployment approach from [finn-quantized-yolo](https://github.com/sefaburakokcu/finn-quantized-yolo).

---

## Overview

A quantized YOLOv5 model was trained on a 10-class VOC subset and the full FINN build pipeline was run through RTL simulation and out-of-context synthesis. 

**Key results from FINN build:**
- All 19 FINN build steps ran including RTL simulation
- RTL simulation: 346,112 inputs consumed, 25,350 outputs produced
- Clock cycles: 22,053,859
- Target: 10 FPS at 100 MHz on Pynq-Z2

---

## Pipeline

```
YOLOv5 Training (Brevitas QAT)
    ↓  github.com/sefaburakokcu/quantized-yolov5
ONNX Export (best.finn.onnx)
    ↓
FINN Build Pipeline (19 steps)
    ↓
RTL Simulation & Verification  ✅
    ↓
Vivado Synthesis (Pynq-Z2)     ⚠️ FIFO depth issue (fix documented below)
    ↓
Deployment & Inference
```

---

## Requirements

### Host Machine
- [FINN](https://github.com/Xilinx/finn) (Docker-based)
- Vivado 2022.1

### Pynq-Z2 Board
- Pynq image v2.7 (required — newer images use Python 3.10 which is incompatible with available armv7l PyTorch wheels)
- Python 3.8
- PyTorch 1.8.1 (armv7l)

---

## Installation

### Host (FINN Docker)

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

PyTorch armv7 wheels: [KumaTea/pytorch-arm](https://github.com/KumaTea/pytorch-arm/releases/tag/v1.8.1)

> **Note:** The board has no internet access by default. Download wheels on your host and transfer via `scp`.

---

## Usage

### 1. Train the Quantized Model

Use [quantized-yolov5](https://github.com/sefaburakokcu/quantized-yolov5) to train and export `best.finn.onnx`.

### 2. Run the FINN Build

Open `build.ipynb` inside the FINN Docker Jupyter environment and run all cells:

```python
import os
os.environ["FINN_BUILD_DIR"] = "/home/user/finn_build"  # use persistent dir, not /tmp
os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)

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
    verbose=True,
    save_intermediate_models=True
)
```

> **Important:** Run the build inside a `tmux` session to prevent interruption:
> ```bash
> tmux new -s finn_build
> ```

### 3. Deploy to Pynq-Z2

```bash
scp -r output_yolov5_pynq/deploy/ xilinx@<PYNQ_IP>:/home/xilinx/finn_deploy/
```

Then open `inference.ipynb` at `http://<PYNQ_IP>:9090`.

---

## FINN Build Steps

| # | Step | Status |
|---|------|--------|
| 1 | step_qonnx_to_finn | ✅ |
| 2 | step_tidy_up | ✅ |
| 3 | step_streamline | ✅ |
| 4 | step_convert_to_hw | ✅ |
| 5 | step_create_dataflow_partition | ✅ |
| 6 | step_specialize_layers | ✅ |
| 7 | step_target_fps_parallelization | ✅ |
| 8 | step_apply_folding_config | ✅ |
| 9 | step_minimize_bit_width | ✅ |
| 10 | step_generate_estimate_reports | ✅ |
| 11 | step_hw_codegen | ✅ |
| 12 | step_hw_ipgen | ✅ |
| 13 | step_set_fifo_depths | ✅ |
| 14 | step_create_stitched_ip | ✅ |
| 15 | step_measure_rtlsim_performance | ✅ |
| 16 | step_out_of_context_synthesis | ✅ |
| 17 | step_synthesize_bitfile | ⚠️ See known issues |
| 18 | step_make_pynq_driver | ⚠️ |
| 19 | step_deployment_package | ⚠️ |

---

## Known Issues
Bitfile generation was not completed due to a FIFO depth compatibility issue between FINN's auto-sizing and Vivado's IP FIFO limit on the XC7Z020 — a fix is documented in the Known Issues section.

### FIFO Depth Exceeds Vivado IP Limit

FINN's automatic FIFO sizing set `StreamingFIFO_rtl_3` to depth `524288`, but the Vivado `axis_data_fifo` IP only supports up to `32768` on the XC7Z020.

**Error:**
```
ERROR: [IP_Flow 19-3461] Value '524288' is out of the range for parameter 
'FIFO depth(FIFO_DEPTH)'. Valid values are: 16, 32, 64, ..., 32768
```

**Fix:** Patch the FINN source to cap the depth:

In `finn/src/finn/custom_op/fpgadataflow/rtl/streamingfifo_rtl.py`, find the line with `CONFIG.FIFO_DEPTH` and change:
```python
# Before
"[get_bd_cells /%s/fifo]" % (depth, node_name)

# After
"[get_bd_cells /%s/fifo]" % (min(depth, 32768), node_name)
```

Then rerun from `step_create_stitched_ip`.

### Build Directory Permissions in Docker

The default `FINN_BUILD_DIR` of `/tmp/finn_dev_<user>` may be owned by root after a container restart, causing permission errors. Fix from outside the container:

```bash
docker exec -u root <container_name> chown -R <user>:<user> /tmp/finn_dev_<user>
```

Or avoid the issue entirely by pointing `FINN_BUILD_DIR` to a persistent, user-owned directory instead of `/tmp`.

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
│   └── intermediate_models/        # Per-step ONNX checkpoints
└── README.md
```

---

## References

- [FINN Framework](https://github.com/Xilinx/finn)
- [Brevitas QAT Library](https://github.com/Xilinx/brevitas)
- [quantized-yolov5 Training Repo](https://github.com/sefaburakokcu/quantized-yolov5)
- [finn-quantized-yolo Deployment Repo](https://github.com/sefaburakokcu/finn-quantized-yolo)
- [LPYOLO Paper](https://arxiv.org/abs/2207.10482)
- [End-to-End Deployment Guide](https://medium.com/@bestamigunay1/end-to-end-deployment-of-lpyolo-low-precision-yolo-for-face-detection-on-fpga-13c3284ed14b)
- [Pynq-Z2 Board](http://www.pynq.io/board.html)
