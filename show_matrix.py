from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import main

y_test, y_pred = main.get_y()
model = main.get_model()

# Crear matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicción")
plt.ylabel("Verdadero")
plt.title("Matriz de Confusión")
plt.show()
