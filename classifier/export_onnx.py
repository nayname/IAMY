from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

SRC       = "artifacts/deberta_lora"   # your fine-tuned HF checkpoint (PyTorch)
ONNX_DIR  = Path("artifacts/onnx")     # will contain model.onnx + tokenizer files
INT8_DIR  = Path("artifacts/onnx-int8")

# 1) Export to ONNX
tokenizer = AutoTokenizer.from_pretrained(SRC)
ort_model = ORTModelForSequenceClassification.from_pretrained(SRC, export=True)
ort_model.save_pretrained(ONNX_DIR)
tokenizer.save_pretrained(ONNX_DIR)

# 2) Quantize the ONNX model (dynamic INT8 is portable and safe on AVX2 CPUs)
quantizer = ORTQuantizer.from_pretrained(ONNX_DIR)
qcfg = AutoQuantizationConfig.dynamic()  # use avx512_vnni only if your CPU supports it
quantizer.quantize(save_dir=INT8_DIR, quantization_config=qcfg)

print("ONNX saved to:", ONNX_DIR.resolve())
print("INT8 quantized ONNX saved to:", INT8_DIR.resolve())