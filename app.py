import numpy as np
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

model = tf.keras.models.load_model("modelo_mnist.keras")
MAX_DIGITS = 4


def image_on_white(image):
    if image is None:
        raise gr.Error("Subí o pegá una imagen.")
    return Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA")).convert("RGB")


def extract_foreground(image):
    image = image_on_white(image)
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
    """Return model inputs, a binary mask, and the bounds of detected digits."""
    pixels = extract_foreground(image)
    mask = pixels > 0.15
    ys, xs = np.where(mask)
    if not len(xs):
        raise gr.Error("No encontré dígitos separados del fondo.")
    top, bottom = ys.min(), ys.max() + 1
    left, right = xs.min(), xs.max() + 1
    cropped_pixels = pixels[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]
    segments = _column_segments(cropped_mask.astype(np.float32))
    if not segments:
        raise gr.Error("No pude separar los dígitos. Escribilos más grandes y con trazos más gruesos.")
    if len(segments) > MAX_DIGITS:
        raise gr.Error(f"Podés ingresar como máximo {MAX_DIGITS} dígitos por imagen. Separá el número en imágenes más pequeñas.")

    digits = []
    bounds = []
    for start, end in segments:
        segment_mask = cropped_mask[:, start:end]
        segment_ys = np.where(segment_mask)[0]
        if not len(segment_ys):
            continue
        digits.append(normalize_digit(cropped_pixels[:, start:end]))
        bounds.append((left + start, top + segment_ys.min(), left + end - 1, top + segment_ys.max()))

    if not digits:
        raise gr.Error("No pude separar los dígitos. Escribilos más grandes y con trazos más gruesos.")
    return np.stack(digits), Image.fromarray(mask.astype(np.uint8) * 255), bounds


def predict(image):
    digits_input, binary_mask, bounds = preprocess(image)
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

    marked_regions = image_on_white(image)
    draw = ImageDraw.Draw(marked_regions)
    line_width = max(2, round(max(marked_regions.size) / 250))
    for index, (left, top, right, bottom) in enumerate(bounds, start=1):
        draw.rectangle((left, top, right, bottom), outline="red", width=line_width)
        draw.text((left, max(0, top - 14)), str(index), fill="red", stroke_width=1, stroke_fill="white")

    preview = np.zeros((28, 28 * len(digits_input) + 4 * (len(digits_input) - 1)), dtype=np.float32)
    for index, digit in enumerate(digits_input):
        start = index * 32
        preview[:, start:start + 28] = digit
    preview = Image.fromarray((preview * 255).astype(np.uint8)).resize(
        (preview.shape[1] * 8, preview.shape[0] * 8), Image.Resampling.NEAREST
    )
    return digits, fig, binary_mask, marked_regions, preview

with gr.Blocks() as demo:
    gr.Markdown("# Clasificador MNIST")
    gr.Markdown(f"Escribí hasta {MAX_DIGITS} dígitos grandes, con trazos claros y separados. La máscara muestra los trazos detectados; los cuadros rojos indican cada recorte. La entrada al modelo muestra cada recorte centrado en 28×28 píxeles.")
    upload = gr.Image(label="Subí o pegá una imagen", type="pil", sources=["upload", "clipboard"])
    button = gr.Button("Predecir", variant="primary")
    number = gr.Textbox(label="Número predicho", text_align="center", scale=1)
    chart = gr.Plot(label="Probabilidades")
    binary_preview = gr.Image(label="Imagen en blanco y negro (trazo detectado en blanco)")
    regions_preview = gr.Image(label="Partes analizadas (cuadros rojos)")
    model_preview = gr.Image(label="Entrada exacta al modelo (28×28 por dígito)")
    button.click(predict, upload, [number, chart, binary_preview, regions_preview, model_preview])

if __name__ == "__main__":
    demo.launch()
