import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Cargar los datos de productos desde el archivo CSV.
data = pd.read_csv("products.csv")

# Separar las características de entrada de la etiqueta que queremos predecir.
features = data[["feature1", "feature2", "feature3"]]
labels = data["label"]

# Dividir los datos en conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Crear y entrenar el modelo de vecinos más cercanos.
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluar el modelo con el conjunto de prueba e imprimir la exactitud.
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Predecir la etiqueta recomendada para las características de un producto.
def recommend(product_features):
    product_array = np.asarray(product_features, dtype=float).reshape(1, -1)
    return model.predict(product_array)[0]


# Ejemplo de uso de la función de recomendación.
if __name__ == "__main__":
    example_product = [1.0, 2.0, 3.0]
    recommendation = recommend(example_product)
    print("Recommended Product:", recommendation)