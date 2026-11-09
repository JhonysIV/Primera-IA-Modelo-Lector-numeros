# Cleanup after the MNIST/EMNIST activity

Run these commands from this project directory when you are done and no longer need the app or training files.

## Remove installed Python packages and environments

```bash
rm -rf .venv
rm -rf /home/johnys/tensorflow_datasets
rm -rf /home/johnys/.cache/pip
```

- `.venv` contains TensorFlow, Gradio, TensorFlow Datasets, and all project Python dependencies.
- `/home/johnys/tensorflow_datasets` contains the downloaded EMNIST dataset.
- `/home/johnys/.cache/pip` is only the package-download cache. Remove it only if you want to clear cached packages for all Python projects.

## Optionally remove generated model and temporary images

```bash
rm -f modelo_mnist.keras debug_7.png debug_emnist.png
```

## Optionally remove the whole activity folder

Only run this if you want to delete the entire project, including the notebook, application, model, and images:

```bash
cd ..
rm -rf algo
```
