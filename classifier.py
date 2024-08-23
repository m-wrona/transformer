import torch
import tensorflow
from transformers import pipeline

x = torch.rand(5, 3)
print(f"Torch: {x}")

print(f"Tensorflow: {tensorflow.reduce_sum(tensorflow.random.normal([1000, 1000]))}")

classifier = pipeline("sentiment-analysis")
results = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)

print(f"Classifier: {results}")