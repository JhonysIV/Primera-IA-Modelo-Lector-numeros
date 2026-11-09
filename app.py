import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model("modelo_mnist.keras")


def extract_foreground(image):
    if image is None:
        raise gr.Error("Subí o pegá una imagen.")
    image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA")).convert("RGB")
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    border = np.concatenate([grayscale[0], grayscale[-1], grayscale[:, 0], grayscale[:, -1]])
    background = np.median(border)
    return np.abs(grayscale - background) / 255.0


def normalize_digit(digit):
    ys, xs = np.where(digit > 0.15)
    if not len(xs):
        raise gr.Error("No encontré dígitos separados del fondo.")
    digit = digit[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = 20 / max(digit.shape)
    size = tuple(max(1, round(side * scale)) for side in digit.shape[::-1])
    digit = np.asarray(Image.fromarray((digit * 255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS)) / 255.0
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - digit.shape[0]) // 2, (28 - digit.shape[1]) // 2
    canvas[top:top + digit.shape[0], left:left + digit.shape[1]] = digit
    return canvas


def _column_segments(pixels):
    column_counts = pixels.sum(axis=0)
    active = column_counts > 0
    edges = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    segments = [(start, end) for start, end in zip(starts, ends) if end - start >= 2]

    max_gap = max(1, round(pixels.shape[0] * 0.06))
    merged = []
    for start, end in segments:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    result = []
    for start, end in merged:
        while end - start > pixels.shape[0] * 1.35:
            counts = column_counts[start:end]
            margin = max(1, round(len(counts) * 0.2))
            cut = start + margin + int(np.argmin(counts[margin:-margin]))
            if cut - start < pixels.shape[0] * 0.18 or end - cut < pixels.shape[0] * 0.18:
                break
            result.append((start, cut))
            start = cut
        result.append((start, end))
    return result


def preprocess(image):
    """Convert an image of separated digits to one MNIST-sized image per digit."""
    pixels = extract_foreground(image)
    mask = pixels > 0.15
    ys, xs = np.where(mask)
    if not len(xs):
        raise gr.Error("No encontré dígitos separados del fondo.")
    pixels = pixels[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    segments = _column_segments(mask.astype(np.float32))
    digits = [normalize_digit(pixels[:, start:end]) for start, end in segments]
    return np.stack(digits)


def predict(image):
    digits_input = preprocess(image)
    probabilities = model.predict(digits_input, verbose=0)
    digits = "".join(map(str, probabilities.argmax(axis=1)))
    fig, ax = plt.subplots(figsize=(7, 3))
    positions = np.arange(10)
    width = 0.8 / len(probabilities)
    for index, values in enumerate(probabilities):
        ax.bar(positions - 0.4 + width / 2 + index * width, values, width=width, label=f"Posición {index + 1}")
    ax.set(xticks=positions, ylim=(0, 1), xlabel="Dígito", ylabel="Probabilidad")
    if len(probabilities) > 1:
        ax.legend()
    preview = np.zeros((28, 28 * len(digits_input) + 4 * (len(digits_input) - 1)), dtype=np.float32)
    for index, digit in enumerate(digits_input):
        start = index * 32
        preview[:, start:start + 28] = digit
    preview = Image.fromarray((preview * 255).astype(np.uint8))
    return digits, fig, preview

with gr.Blocks() as demo:
    gr.Markdown("# Clasificador MNIST")
    gr.Markdown("Escribí un número grande, con trazos claros y separados. La vista previa muestra lo que recibe el modelo; si el dígito no se ve bien ahí, volvé a subir la imagen.")
    upload = gr.Image(label="Subí o pegá una imagen", type="pil", sources=["upload", "clipboard"])
    button = gr.Button("Predecir", variant="primary")
    number = gr.Textbox(label="Número predicho", text_align="center", scale=1)
    chart = gr.Plot(label="Probabilidades")
    preview = gr.Image(label="Vista previa procesada")
    button.click(predict, upload, [number, chart, preview])

if __name__ == "__main__":
    demo.launch()
