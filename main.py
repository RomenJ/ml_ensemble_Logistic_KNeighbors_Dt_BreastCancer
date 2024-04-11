import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filename):
    """Cargar el conjunto de datos desde el archivo CSV."""
    try:
        data = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{filename}'")
        return None
    data['diagNum'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
    return data

def print_validation_metrics(model, X_test, y_test):
    """Mostrar los valores de validez por consola."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f'Accuracy: {accuracy:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}')

def plot_contour_plots(classifiers, X_scaled, data):
    """Gráficos de contorno para cada clasificador."""
    plt.figure(figsize=(18, 5))
    for i, (clf_name, clf) in enumerate(classifiers + [('Voting Classifier', vc)], start=1):
        plt.subplot(1, 4, i)
        clf.fit(X_train, y_train)
        h = 0.02  # Step size in the mesh
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis' if clf_name != 'Decision Tree' else 'cividis')
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=data['diagNum'], cmap='viridis', s=50, edgecolors='k', alpha=0.8)
        plt.title(clf_name)
        plt.savefig(f'{clf_name}.jpg')
        plt.xlabel('radius_mean')
        plt.ylabel('texture_mean')

    plt.tight_layout()
    plt.savefig('ContourPlots.png')
    plt.show()

def plot_metrics_plots(classifiers, X_train, y_train, X_test, y_test):
    """Gráficos de métricas para cada modelo."""
    plt.figure(figsize=(18, 5))
    metrics = ['Accuracy', 'Sensitivity', 'Specificity']
    for i, metric in enumerate(metrics, start=1):
        plt.subplot(1, 3, i)
        for clf_name, clf in classifiers + [('Voting Classifier', vc)]:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            if metric == 'Accuracy':
                metric_score = accuracy_score(y_test, y_pred)
            else:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                if metric == 'Sensitivity':
                    metric_score = tp / (tp + fn)
                elif metric == 'Specificity':
                    metric_score = tn / (tn + fp)
            plt.bar(clf_name, metric_score, label=clf_name)
        plt.title(f'{metric} of Models')
        plt.xlabel('Models')
        plt.ylim(0.8, 1.0)  # Establecer el rango de los valores de Y
        plt.xticks(rotation=45)  # Rotar las etiquetas del eje X
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout()
    plt.savefig('MetricsPlots.png')
    plt.show()

def plot_cross_validation(classifiers, X_train, y_train):
    """Gráfico de validación cruzada."""
    plt.figure(figsize=(10, 6))
    cv_scores = {}
    line_styles = ['-', '--', '-.', ':', '-']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Colores amigables para daltónicos
    for i, (clf_name, clf) in enumerate(classifiers + [('Voting Classifier', vc)]):
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        cv_scores[clf_name] = scores
        sns.distplot(scores, hist=False, kde_kws={'shade': True, 'linestyle': line_styles[i], 'color': colors[i]}, label=clf_name)

    plt.title('Cross-Validation Scores')
    plt.xlabel('Accuracy')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('CrossValidation.png')
    plt.show()

def find_best_model(classifiers, X_train, y_train, X_test, y_test):
    """Encontrar el mejor modelo y mostrar sus parámetros."""
    best_model = None
    best_accuracy = 0.0
    for clf_name, clf in classifiers:
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf_name
    return best_model

# Cargar el conjunto de datos desde el archivo CSV
data = load_data('BreastCancer.csv')
if data is not None:
    # Seleccionar características para el análisis
    selected_features = data[['radius_mean', 'texture_mean']].copy()  # Usaremos solo dos características para la visualización

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(selected_features)
    y = data['diagNum']  # Variable objetivo

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Instanciar modelos
    lr = LogisticRegression(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=27)
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=42)

    # Lista de clasificadores
    classifiers = [('Logistic Regression', lr),
                   ('K-nearest Neighbours', knn),
                   ('Decision Tree', dt)]

    # Instanciar el clasificador de votación
    vc = VotingClassifier(estimators=classifiers, voting='soft')

    plot_contour_plots(classifiers, X_scaled, data)
    plot_metrics_plots(classifiers, X_train, y_train, X_test, y_test)
    plot_cross_validation(classifiers, X_train, y_train)

    # Imprimir los valores de validez por consola
    print("Validation metrics:")
    for clf_name, clf in classifiers + [('Voting Classifier', vc)]:
        print(f"{clf_name}:")
        print_validation_metrics(clf, X_test, y_test)
        print()

    # Mostrar el mejor modelo
    best_model = find_best_model(classifiers, X_train, y_train, X_test, y_test)
    print(f"Overall, the best model is '{best_model}'.")
