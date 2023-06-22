# Trabajo Práctico 4 - Implementaciones de Kohonen, Oja y Hopfield

## Dependencias

- Python **>= 3.11**
- Pipenv

## Set up

Primero se deben descargar las dependencias a usar en el programa. Para ello podemos hacer uso de los archivos _Pipfile_ y _Pipfile.lock_ provistos, que ya las tienen detalladas. Para usarlos se debe correr en la carpeta del TP4:

```bash
$> pipenv shell
$> pipenv install
```

Esto creará un nuevo entorno virtual, en el que se instalarán las dependencias a usar, que luego se borrarán una vez se cierre el entorno.

**NOTA:** Previo a la instalación se debe tener descargado **python** y **pipenv**, pero se omite dicho paso en esta instalación.

## Cómo Correr

```bash
python ej1_1.py
python ej1_2.py
python ej2.py
```
donde ej1_1 es de Kohonen
ej1_2 es de Oja
ej2 es de Hopfield

## Archivo de Configuración:

### Configuraciones Basicas

**Nota: opción_a | opción_b | opción_c representa un parámetro que puede tomar únicamente esas opciones**

```json5
{
  "kohonen": {
    "grid_dimension": 4,
    "radius": 1.8,
    "learning_rate": 0.9,
    "epochs": 500,
    "random_weights": false | true
  },
  "oja": {
    "learning_rate": 0.0001,
    "epochs": 1000
  },
  "hopfield": {
    "epochs": 1000
  }
}
```

### Archivos de salida

Los archivos de salida son iamgens png, que se encuentran en la carpeta `outputs`.
