# Algoritmo Genético para CartPole

Este proyecto implementa un algoritmo genético para resolver el problema de CartPole usando Gymnasium, con soporte opcional para GPU usando CUDA.

## Descripción

El problema de CartPole consiste en mantener un polo en equilibrio sobre un carrito móvil. El objetivo es mantener el polo en posición vertical el mayor tiempo posible.

### Características del entorno
- 4 observaciones (estados):
  - Posición del carrito
  - Velocidad del carrito
  - Ángulo del polo
  - Velocidad angular del polo
- 2 acciones posibles:
  - 0: Mover el carrito a la izquierda
  - 1: Mover el carrito a la derecha

### Criterios de éxito
El episodio termina cuando:
- El polo se inclina más de 15 grados
- El carrito se mueve más de 2.4 unidades desde el centro
- El episodio dura más de 500 pasos de tiempo

## Requisitos

### Requisitos Generales
- Python 3.8+
- pip (gestor de paquetes de Python)

### Requisitos para GPU (Opcional)
- GPU NVIDIA compatible con CUDA
- Drivers NVIDIA actualizados
- CUDA Toolkit (versión compatible con PyTorch)
- Mínimo 4GB de VRAM
- 8GB de RAM del sistema

## Instalación

### Windows

1. **Clonar el repositorio**:
```bash
# Clonar el repositorio
git https://github.com/GiomarMC/Application_of_genetic_algorithms.git
cd algoritmos-geneticos
```

2. **Crear entorno virtual**:
```bash
# Crear entorno virtual
python -m venv venv
```

3. **Activar entorno virtual**:
```bash
# Activar entorno virtual
.\venv\Scripts\activate
```

4. **Instalar dependencias** (con el entorno virtual activado):
```bash
# Instalar PyTorch con soporte CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar resto de dependencias
pip install -r requirements.txt
```

### Linux

1. **Clonar el repositorio**:
```bash
# Clonar el repositorio
git https://github.com/GiomarMC/Application_of_genetic_algorithms.git
cd algoritmos-geneticos
```

2. **Crear entorno virtual**:
```bash
# Crear entorno virtual
python3 -m venv venv
```

3. **Activar entorno virtual**:
```bash
# Activar entorno virtual
source venv/bin/activate
```

4. **Instalar dependencias** (con el entorno virtual activado):
```bash
# Instalar PyTorch con soporte CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar resto de dependencias
pip install -r requirements.txt
```

## Uso

1. **Activar el entorno virtual** (si no está activado):
```bash
# Windows
.\venv\Scripts\activate

# Linux
source venv/bin/activate
```

2. **Ejecutar el programa**:
```bash
python main.py
```

## Estructura del Proyecto

```
.
├── main.py              # Punto de entrada principal
├── genetic_algorithm.py # Implementación del algoritmo genético
├── neural_network.py    # Implementación de la red neuronal
├── requirements.txt     # Dependencias del proyecto
└── outputs/            # Directorio para guardar resultados
    ├── learning_curve.png
    └── best_individual_*.npy
```

## Características del Algoritmo

- **Paralelización Automática**: Detecta automáticamente si hay GPU disponible y usa CUDA cuando es posible
- **Early Stopping**: Detiene el entrenamiento si no hay mejora significativa
- **Mantenimiento de Diversidad**: Implementa mecanismos para mantener la diversidad de la población
- **Elitismo**: Preserva los mejores individuos entre generaciones
- **Mutación Adaptativa**: Tasa de mutación que se ajusta según la diversidad de la población

## Verificación de GPU

Para verificar si CUDA está disponible en tu sistema:
```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Número de GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU actual: {torch.cuda.get_device_name(0)}")
```

## Licencia

Este proyecto está bajo la Licencia MIT. 