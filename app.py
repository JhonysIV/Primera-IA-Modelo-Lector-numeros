import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model("modelo_mnist.keras")

def preprocess(image):
    """Converts PNG/JPG or canvas input to MNIST shape (1, 28, 28)."""
    if isinstance(image, dict):
        image = image.get("composite") or image.get("background")
    if image is None:
        raise gr.Error("Subí o dibujá una imagen.")
    image = Image.alpha_composite(Image.new("RGBA", image.size, "black"), image.convert("RGBA"))
    pixels = (255.0 - np.asarray(image.convert("L"), dtype=np.float32)) / 255.0
    ys, xs = np.where(pixels > 0.15)
    if not len(xs):
        raise gr.Error("No encontré un dígito oscuro sobre un fondo claro.")
    digit = pixels[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = 20 / max(digit.shape)
    size = tuple(max(1, round(side * scale)) for side in digit.shape[::-1])
    digit = np.asarray(Image.fromarray((digit * 255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS)) / 255.0
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - digit.shape[0]) // 2, (28 - digit.shape[1]) // 2
    canvas[top:top + digit.shape[0], left:left + digit.shape[1]] = digit
    return canvas.reshape(1, 28, 28)

def predict(image):
    probabilities = model.predict(preprocess(image), verbose=0)[0]
    digit = int(np.argmax(probabilities))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(10), probabilities)
    ax.set(xticks=range(10), ylim=(0, 1), xlabel="Dígito", ylabel="Probabilidad")
    return str(digit), fig

with gr.Blocks() as demo:
    gr.Markdown("# Clasificador MNIST")
    upload = gr.Image(label="Subí o pegá una imagen", type="pil", sources=["upload", "clipboard"])
    button = gr.Button("Predecir", variant="primary")
    number = gr.Textbox(label="Número predicho", text_align="center", scale=1)
    chart = gr.Plot(label="Probabilidades")
    button.click(predict, upload, [number, chart])

if __name__ == "__main__":
    demo.launch()
