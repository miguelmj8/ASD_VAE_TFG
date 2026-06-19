# ASD VAE Project

## Descripción

Proyecto de Autoencoder con distintas variantes para análisis de datos ASD.

El código base utilizado proviene inicialmente de los siguientes repositorios:

- https://github.com/daisukelab/dcase2020_task2_variants/tree/master/2vae_pytorch
- https://github.com/Kota-Dohi/dcase2022_task2_baseline_ae

---

## Requisitos

- Python 3.11 o superior
- Dependencias incluidas en `requirements.txt`

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```text
.
├── train.py
├── test.py
├── trainCNN.py
├── testCNN.py
├── trainCNNClass.py
├── testCNNClass.py
├── Inference.py
├── OneClassSVM.py
├── ensemble.py
├── da.py
├── common.py
├── tsne_view.py
├── model/
│   └── vae_model.py
│   └── cnn_vae.py
│   └── cnn_vaeClass.py
├── model_output/
├── results/
└── data/
    ├── data/
    │   └── TipoDeMaquina/
    │       ├── train/
    │       └── test/
    └── Features/
        └── melspecTamaño/
            └── TipoDeMaquina/
                ├── train/
                └── test/
```

---

## Preparación de los Datos

1. Coloca los archivos `.wav` en:

```text
data/data/TipoDeMaquina/
```

2. Los espectrogramas calculados se almacenarán automáticamente en:

```text
data/Features/melspecTamaño/TipoDeMaquina/
```

---

## Entrenamiento

### Modelos Lineales

Entrenamiento de AE o VAE con capas lineales:

```bash
python train.py
```

### Modelos Convolucionales

Entrenamiento de AE o VAE con capas convolucionales:

```bash
python trainCNN.py
```

### Modelos Híbridos con Clasificación

Entrenamiento de AE o VAE convolucional híbrido con clasificación por tipo de máquina en el espacio latente:

```bash
python trainCNNClass.py
```

---

## Evaluación y Test

### Evaluación principal

```bash
python test.py
python testCNN.py
python testCNNClass.py
```

### Evaluación con métodos adicionales

```bash
python OneClassSVM.py
```

Este script permite evaluar utilizando otros métodos además de clasificadores de una sola clase.

---

## Inferencia

```bash
python Inference.py
```

Permite:

- Obtener la predicción de un audio individual
- Visualizar la reconstrucción generada por el modelo

---

## Scripts Auxiliares

### `common.py`

Contiene funciones comunes utilizadas por distintos scripts.

### `tsne_view.py`

Genera visualizaciones t-SNE para el modelo especificado en el fichero de parámetros seleccionado.

### `ensemble.py`

Construye automáticamente un modelo *ensemble* óptimo.

### `da.py`

Modela el espacio latente utilizando BGM (*Bayesian Gaussian Mixture*) y:

- Genera muestras latentes
- Reconstruye dichas muestras
- Almacena los resultados generados

---

## Opciones Adicionales

### Aumento de Datos

Para entrenar utilizando *data augmentation*:

```bash
python train.py -d
```

### Evaluación con conjunto de evaluación

```bash
python test.py -e
```

### Evaluación por resustitución

Evalúa utilizando los propios datos de entrenamiento:

```bash
python test.py -r
```

---

## Selección entre AE y VAE

Para utilizar el modelo como:

- Autoencoder (`AE`)
- Variational Autoencoder (`VAE`)

es necesario modificar la variable:

```python
vae = True  # o False
```

en las primeras líneas de los siguientes scripts:

- `train*.py`
- `test*.py`
- `OneClassSVM.py`

---

## Resultados

Los resultados se almacenan en la carpeta:

```text
results/
```

### Estructura recomendada

```text
./results/
├── CNN o Lineal/
│   ├── VAE o AE/
│   │   ├── Class o NoClass/
│   │   │   └── ParámetrosTrain/
│   │   │       └── ParámetrosModeloYFeatures/
```

---

## Modelos Guardados

Los modelos entrenados se almacenan en:

```text
model_output/
```