import urllib.request
import os

# Create the folder where the model file will live
os.makedirs("model_files", exist_ok=True)

# The live community URL for the 26.9MB INT8 quantized MobileBERT model
url = "https://huggingface.co/onnx-community/mobilebert-uncased-squad-v2-ONNX/resolve/main/onnx/model_quantized.onnx"

print("Downloading Quantized MobileBERT Model from Hugging Face...")
try:
    urllib.request.urlretrieve(url, "model_files/model.onnx")
    print("Download complete! Model saved perfectly in model_files/model.onnx")
except Exception as e:
    print(f"An error occurred during download: {e}")