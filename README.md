# ASD VAE Project

## Descripción
Proyecto de Variational Autoencoder (VAE) para análisis de datos ASD.
Utilizamos codigo extraido principalmete de
- https://github.com/daisukelab/dcase2020_task2_variants/tree/master/2vae_pytorch
- https://github.com/Kota-Dohi/dcase2022_task2_baseline_ae

## Instalación
```bash
pip install -r requirements.txt
```

## Uso
1. Prepara tu conjunto de datos en la carpeta `data/`
2. Ejecuta el entrenamiento:
    ```bash
    python train.py
    ```
3. Ejecuta el test
     ```bash
    python test.py
    ```

4. Para utilizar como autoencoder AE (no VAE) es necesario comentar algunas lineas y descomentar otras en vae_model.py, train.py y test.py

## Estructura
- `train.py` - Script de entrenamiento
- `test.py` - Script de inferencia y evaluacion, guarda los resultados en results/
- `results/` - Carpeta con los resultados
- `../data/` - Carpeta de datos
    - `../data/data/` - Carpeta de datos .wav, contiene una carpeta para cada tipo de maquina, y a su vez cada tipo de maquina contiene una carpeta train (solo datos normales) y otra test
    - `../data/Features/melspec__` - Carpeta de datos .npy, contiene una carpeta para cada tipo de maquina, y a su vez cada tipo de maquina contiene una carpeta train (solo datos normales) y otra test
- `model/` - Carpeta donde se encuentra vae_model.py, donde se define el modelo
- `model_output/` - Modelos guardados

## Requisitos
- requirements.txt
- Python 3.11 al menos


