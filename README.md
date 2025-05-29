# Algoritmo Genético para Control de Agente

Este proyecto implementa un Algoritmo Genético (AG) para entrenar una red neuronal que controla un agente en un entorno de simulación. El algoritmo utiliza una población de individuos, donde cada individuo representa los pesos de una red neuronal simple.

## Prerrequisitos

### Windows
- Python 3.8 o superior
- Visual Studio Build Tools (para compilar algunas dependencias)
- Git (opcional, para clonar el repositorio)

### Linux
- Python 3.8 o superior
- build-essential
- python3-dev
- git (opcional)

## Instalación

1. Clonar el repositorio (o descargar los archivos):
```bash
git clone <url-del-repositorio>
cd <nombre-del-directorio>
```

2. Crear y activar entorno virtual:

### Windows
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
.\venv\Scripts\activate
```

### Linux
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Estructura del Proyecto
```
.
├── genetic_algorithm.py    # Implementación del algoritmo genético
├── main.py                # Script principal de ejecución
├── visualize.py           # Visualización de resultados
├── requirements.txt       # Dependencias del proyecto
├── .gitignore            # Archivos y carpetas ignorados por git
├── outputs/              # Directorio para archivos generados
│   ├── learning_curve.png # Gráfica de aprendizaje
│   └── *.npy             # Archivos de pesos guardados
└── README.md             # Este archivo
```

## Uso

1. Asegúrate de tener el entorno virtual activado
2. Ejecuta el script principal:
```python
python main.py
```

## Características
- Red neuronal de dos capas (feed-forward)
- Selección por torneo
- Cruce de punto único
- Mutación gaussiana
- Elitismo
- Visualización de la curva de aprendizaje
- Guardado automático de los mejores individuos

## Parámetros Configurables
- Tamaño de población
- Número de generaciones
- Tasa de mutación
- Tamaño de la capa de entrada
- Tamaño de la capa de salida
- Tamaño de la capa oculta

## Notas
- Los resultados del entrenamiento se guardan en la carpeta `outputs/`
- La gráfica de aprendizaje se guarda como `learning_curve.png`
- Los mejores individuos se guardan como archivos `.npy`

## Solución de Problemas

### Windows
Si encuentras errores al instalar las dependencias:
1. Asegúrate de tener Visual Studio Build Tools instalado
2. Ejecuta el comando como administrador
3. Actualiza pip: `python -m pip install --upgrade pip`

### Linux
Si encuentras errores al instalar las dependencias:
1. Instala los paquetes de desarrollo necesarios:
```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
```
2. Actualiza pip: `python3 -m pip install --upgrade pip` 