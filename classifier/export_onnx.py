from optimum.onnxruntime import ORTQuantizer, ORTOptimizer
from optimum.onnxruntime.config import AutoQuantizationConfig
from transformers import AutoTokenizer
from pathlib import Path

MODEL_PATH = "artifacts/deberta_lora"
OUT_ONNX = Path("artifacts/deberta_intent.onnx")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
ORTQuantizer.from_pretrained(MODEL_PATH).quantize(
    save_dir=OUT_ONNX.parent,
    quantization_config=AutoQuantizationConfig.avx512_vnni(is_static=False),
)