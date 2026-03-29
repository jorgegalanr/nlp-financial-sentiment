# Análisis de Sentimiento Financiero con NLP (FinBERT)

## Descripción del Proyecto
Este proyecto implementa un *pipeline* automatizado de ingeniería de datos e Inteligencia Artificial para el análisis de sentimiento en tiempo real del mercado financiero. Utiliza extracción de noticias web, procesamiento de lenguaje natural (NLP) y visualización de datos para diagnosticar el sesgo (alcista, bajista o neutral) sobre activos específicos.

## Arquitectura y Flujo de Datos
1. **Extracción (Web Scraping):** Conexión a la API de Yahoo Finance mediante `yfinance` para extraer titulares recientes. Implementación de programación defensiva (`.get()`) para tolerar cambios estructurales en los diccionarios de la API origen sin romper el código.
2. **Preprocesamiento y Traducción:** Uso de `deep-translator` para estandarizar las entradas al inglés, superando la barrera del idioma del modelo predictivo.
3. **Inferencia (NLP):** Integración de `ProsusAI/finbert`, un modelo basado en BERT preentrenado con corpus financiero (informes SEC, noticias de mercado), alojado en Hugging Face.
4. **Visualización:** Consolidación de los resultados en un DataFrame de Pandas y renderizado de gráficos de resumen mediante Seaborn y Matplotlib.

## Debilidades del Modelo y Solución Implementada
**El problema:** Modelos especializados como FinBERT presentan un alto rendimiento, pero están fuertemente sesgados hacia el idioma de su entrenamiento (inglés). Ante un titular en español u otro idioma, el modelo no generaliza y devuelve falsos positivos o clasificaciones erróneas, comprometiendo la fiabilidad del análisis.

**La solución:** En lugar de reentrenar un modelo desde cero (costoso e ineficiente para este alcance), se ha implementado un patrón de arquitectura con un paso intermedio. Se intercepta la noticia cruda y se procesa mediante una API de traducción (Google Translate) en tiempo real. El modelo NLP recibe exclusivamente texto estandarizado en inglés, garantizando la precisión del análisis independientemente del origen de la noticia.

## Caso de Estudio: Tesla (TSLA) - Marzo 2026
Durante la fase de validación, se ejecutó el *pipeline* sobre las últimas 10 noticias publicadas del ticker `TSLA`. 

Los resultados mostraron un claro sesgo pesimista en el mercado:
* **0 Noticias Positivas.**
* **5 Noticias Neutrales.**
* **5 Noticias Negativas** (Asociadas a caídas del Dow Jones, volatilidad y bajadas en las acciones del sector EV).

El resultado se consolida en un gráfico de barras que omite dinámicamente las categorías sin valores, mostrando la realidad del entorno de mercado de forma directa para la toma de decisiones.

## Tecnologías Utilizadas
* Python 3
* `transformers` (Hugging Face)
* `yfinance`
* `pandas`
* `deep-translator`
* `matplotlib` & `seaborn`
