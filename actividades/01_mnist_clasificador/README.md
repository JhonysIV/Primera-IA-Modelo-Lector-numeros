# Actividad 1 — Clasificador de números MNIST

**Autor:** Juan Diego Restrepo Hernández
**Curso:** Teoría de Aprendizaje de Máquinas
**Universidad Nacional de Colombia, sede Manizales**

## Descripción

Esta actividad implementa y entrena una red neuronal convolucional para clasificar dígitos escritos a mano usando el conjunto de datos MNIST. También incluye una aplicación interactiva para predecir uno o varios dígitos a partir de una imagen.

## Archivos

Los archivos de la actividad están en [`01_Modelo_Numeros_Mnist/`](./01_Modelo_Numeros_Mnist/):

- [`Untitled8.ipynb`](./01_Modelo_Numeros_Mnist/Untitled8.ipynb): prepara y centra las imágenes, entrena el modelo, evalúa su precisión y guarda el modelo entrenado.
- [`app.py`](./01_Modelo_Numeros_Mnist/app.py): aplicación Gradio para cargar o pegar una imagen y visualizar los dígitos detectados, sus probabilidades y los recortes procesados.
- [`modelo_mnist.keras`](./01_Modelo_Numeros_Mnist/modelo_mnist.keras): modelo entrenado utilizado por la aplicación.

## Ejecución de la aplicación

Instala Python y las dependencias `numpy`, `tensorflow`, `gradio`, `matplotlib` y `Pillow`. Luego, desde la carpeta del modelo, ejecuta:

```bash
cd actividades/01_mnist_clasificador/01_Modelo_Numeros_Mnist
python app.py
```

La aplicación abrirá una interfaz local en el navegador. Se pueden subir o pegar imágenes con hasta cuatro dígitos, escritos grandes y separados.

## Entrenamiento

Abre el notebook en Jupyter o Google Colab y ejecuta sus celdas en orden. El notebook obtiene MNIST mediante Keras y guarda el modelo como `modelo_mnist.keras` cuando la precisión de prueba alcanza al menos 98 %.
