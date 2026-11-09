import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("modelo_mnist.keras")

def preprocess(image):
    """Converts PNG/JPG or canvas input to MNIST shape (1, 28, 28)."""
    if isinstance(image, dict):
        image = image.get("composite") or image.get("background")
    if image is None:
        raise gr.Error("Subí o dibujá una imagen.")
    image = image.convert("L").resize((28, 28))
    pixels = (255.0 - np.asarray(image, dtype=np.float32)) / 255.0
    return pixels.reshape(1, 28, 28)

def predict(image):
    probabilities = model.predict(preprocess(image), verbose=0)[0]
    digit = int(np.argmax(probabilities))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(10), probabilities)
    ax.set(xticks=range(10), ylim=(0, 1), xlabel="Dígito", ylabel="Probabilidad")
    return str(digit), fig

with gr.Blocks() as demo:
    gr.Markdown("# Clasificador MNIST")
    canvas = gr.ImageEditor(
        label="Subí una imagen o dibujá un dígito",
        type="pil",
        sources=["upload", "clipboard"],
        brush=gr.Brush(colors=["#ffffff"]),
    )
    button = gr.Button("Predecir", variant="primary")
    number = gr.Textbox(label="Número predicho", text_align="center", scale=1)
    chart = gr.Plot(label="Probabilidades")
    button.click(predict, canvas, [number, chart])

if __name__ == "__main__":
    demo.launch()
