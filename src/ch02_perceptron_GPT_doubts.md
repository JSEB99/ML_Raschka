# 🧠 Perceptrón - Guía Teórica Completa para Principiantes

Este documento resume los conceptos esenciales del algoritmo **Perceptrón**, con énfasis en teoría, intuición y buenas prácticas. Ideal para estudiantes de machine learning que están dando sus primeros pasos.

---

## 🔍 ¿Qué es un perceptrón?

Un **perceptrón** es un modelo de clasificación binaria supervisado que aprende a separar clases linealmente separables mediante el ajuste de pesos asociados a las características (features) de entrada.

---

## ⚙️ Componentes principales

- `w_` → Vector de pesos (un valor por cada característica).
- `b_` → Bias (umbral), un escalar.
- `eta` → Tasa de aprendizaje.
- `n_iter` → Número de épocas (pasadas sobre el dataset).
- `errors_` → Lista que guarda el número de errores por época.

---

## 🧮 Cálculo del net input y predicción

### 🧾 Fórmula del net input:

\[
z = \mathbf{w} \cdot \mathbf{x} + b
\]

- \( \mathbf{w} \): pesos
- \( \mathbf{x} \): vector de entrada (ejemplo)
- \( b \): bias
- \( \cdot \): producto punto

### 📤 Predicción:

\[
\hat{y} = 
\begin{cases}
1 & \text{si } z \geq 0 \\
0 & \text{si } z < 0
\end{cases}
\]

---

## 🔧 Regla de actualización

Cuando el perceptrón se equivoca, se actualizan los pesos y bias:

\[
\text{update} = \eta \cdot (y - \hat{y})
\]

\[
\mathbf{w} \leftarrow \mathbf{w} + \text{update} \cdot \mathbf{x}
\]
\[
b \leftarrow b + \text{update}
\]

Si predice correctamente, el error es 0 y no hay actualización.

---

## 📉 Conteo de errores

Cada vez que hay un error, se suma 1:

```python
errors += int(update != 0.0)
```

* `int(True)` = 1
* `int(False)` = 0

Así se construye la lista `errors_`, que contiene cuántos errores hubo en cada época.

---

## 📦 Estructura del dataset con `zip(X, y)`

`X`: matriz de forma `(n_samples, n_features)`
`y`: vector de etiquetas de forma `(n_samples,)`


```Python
for xi, target in zip(X, y):
    # xi es una fila de X, target es su etiqueta correspondiente
```

Ejemplo:

```python
X = [[1, 2], [3, 4], [5, 6]]
y = [0, 1, 0]

zip(X, y) → ([1,2], 0), ([3,4], 1), ([5,6], 0)
```

---

## ⚠️ ¿Por qué NO inicializar pesos en cero?

Inicializar con ceros genera problemas:

* El vector de pesos inicial no tiene dirección definida.
* Tras la primera actualización:

  $$
  \mathbf{w} = \eta \cdot \mathbf{x}
  $$

  Es decir, **la dirección de los pesos se alinea con el primer ejemplo** mal clasificado.
* Esto puede causar que el modelo aprenda de forma **sesgada** hacia ese primer ejemplo.

✅ Solución: inicializar `w_` con **valores aleatorios pequeños**, por ejemplo:

```python
rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
```

Esto permite que `w_` tenga una **dirección aleatoria inicial distinta de cero**, ayudando al aprendizaje.

---

## 📐 ¿Qué significa dirección de un vector?

Un vector tiene:

1. Magnitud (longitud)
2. Dirección (hacia dónde apunta)

### El vector **\[0, 0, ..., 0]** (vector cero):

* Tiene magnitud = 0
* ❌ **No tiene dirección**
* ⚠️ Por eso no puedes calcular ángulos ni compararlo con otros vectores

### Si:

$$
\mathbf{w} = \alpha \cdot \mathbf{x}, \quad \alpha \neq 0
$$

Entonces:

* Tienen la **misma dirección** (o la opuesta si $\alpha < 0$)
* Esto se llama **colinealidad**

---

## 🧠 Conclusión final

* El perceptrón es un modelo simple pero poderoso para clasificación binaria.
* Los pesos deben inicializarse con pequeños valores aleatorios, **no ceros**.
* La actualización depende del error entre la etiqueta real y la predicción.
* El modelo ajusta sus pesos en la dirección de los ejemplos mal clasificados.
* El bias permite mover la frontera de decisión sin depender de las features.

---

## 📌 Recomendaciones

* Usar `eta` pequeños como 0.01 o 0.001
* Visualizar la curva de errores (`errors_`) para ver si el modelo aprende
* Normalizar los datos si los valores de las features varían mucho
* Comprobar que `X` y `y` tengan la misma longitud antes de entrenar
