import numpy as np
import matplotlib.pyplot as plt
import tkinter
from keras.models import Sequential
from keras.layers import Dense

# Generación de datos de entrenamiento
X_train = np.array([[0,0]])
y_train = np.array([0])

#Interfaz
ventana = tkinter.Tk()
ventana.geometry("200x300")
ventana.title("IA-P4")

# Función Adaline
class Adaline:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=32)

    def predict(self, X):
        return np.where(self.model.predict(X) > 0.5, 1, 0)


fig, ax = plt.subplots()
def onclick(event):
    global X_train, y_train  
    x, y = event.xdata, event.ydata
    X_train = np.append(X_train, [[x, y]], axis=0)
    if event.button == 1:
        print("Se hizo clic izquierdo")
        y_train = np.append(y_train, 1)
        plt.scatter(x, y, color="blue")
    elif event.button == 3:
        print("Se hizo clic derecho")
        y_train = np.append(y_train, 0)
        plt.scatter(x, y, color="red")
    plt.draw()

cid = fig.canvas.mpl_connect("button_press_event", onclick)

# Función de interpolación KNN
def knn_interpolation(X_train, y_train, X_test, k=3):
    distances = np.sqrt(np.sum((X_train - X_test[:, np.newaxis])**2, axis=2))
    nearest_neighbors = y_train[np.argsort(distances)[:, :k]]
    y_pred = np.round(np.mean(nearest_neighbors, axis=1))
    return y_pred.astype(int)

def calcular():
    global X_train, y_train  
    y_train = np.delete(y_train, 0)
    X_train = np.delete(X_train, 0, axis=0)

    plt.cla()
    print("calcular")
    # Generación de maya de puntos
    x_min, x_max, y_min, y_max = -10, 10, -10, 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .5), np.arange(y_min, y_max, .5))
    X_mesh = np.c_[xx.ravel(), yy.ravel()]

    # Estimación de valores de características de la maya utilizando interpolación KNN
    X_train_norm = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_mesh_norm = (X_mesh - X_train.mean(axis=0)) / X_train.std(axis=0)
    y_mesh = knn_interpolation(X_train_norm, y_train, X_mesh_norm)

    # Clasificación de puntos de la maya utilizando función Adaline
    adaline = Adaline(lr=0.01, epochs=100)
    adaline.fit(X_train_norm, y_train)
    y_mesh = adaline.predict(X_mesh_norm)

    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    plt.contourf(xx, yy, y_mesh.reshape(xx.shape), alpha=0.5, cmap='coolwarm') 
    #puntos
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')

    plt.axhline(y=0, color='black', lw=2)
    #Linea vertical
    plt.axvline(x=0, color='black', lw=2)
    plt.show()
    

def plane():
    #Linea horizontal
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)

    plt.axhline(y=0, color='red', lw=2)
    #Linea vertical
    plt.axvline(x=0, color='red', lw=2)

plane()

entrada1  =tkinter.Button(ventana, text = "CALCULAR",command = calcular, fg= "dark blue", background="#C4F9D1")
entrada1.pack()
entrada1.place(x=60, y=90, height= 40, width=80)
etiqueta = tkinter.Label(ventana, text="PERCEPTRON", fg="dark green", height=3).pack()

plt.show()
ventana.mainloop()
