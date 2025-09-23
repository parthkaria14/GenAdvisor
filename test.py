# import torch

# # Check if CUDA is available
# if torch.cuda.is_available():
#     print(f"CUDA is available! Using device: {torch.cuda.get_device_name(0)}")
# else:
#     print("CUDA is not available. Running on CPU.")
import tensorflow as tf

# Check if TensorFlow can access GPU
if tf.config.list_physical_devices('GPU'):
    print("CUDA is available! TensorFlow is using GPU.")
    # Optional: print the name of the GPU device
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print(f"GPU: {gpu}")
else:
    print("CUDA is not available. TensorFlow is running on CPU.")
