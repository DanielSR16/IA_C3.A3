from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import csv

# Lectura del dataset con la libreria de CSV
def leerDataset():
    tempX = []
    tempY = []
    tempXY = []

    with open('dataset.csv', newline='') as File:  
        reader = csv.reader(File)
        auxLit = []
        tempX = []
        posicion1 = []
        Ydeseada = []
        for row in reader:
            for datos in row:
                newDato  = datos.replace('.','').replace(',','')
                datoCast = float(newDato)
                auxLit.append(datoCast)
 

            # AÃ±adimos las columnas del dataset en una lista individual
            posicion1.append(auxLit[0])
            
            Ydeseada.append(auxLit[1])

            
            auxLit = [] 



        # print(posicion1)
        # print(Ydeseada)
        
    
    # print(posicion1)
    # print(Ydeseada)
    # Se meten los datos en un array para el manejo de los floats
    tempX = np.array(posicion1, dtype=float)
    tempY = np.array(Ydeseada, dtype=float)

    # Se juntan los arreglos de X y Y
    tempXY = [tempX, tempY]
    # print(tempXY[1])
    # print(tempX)
    # print(tempY)
    # Y se retorna
    
    return tempXY

def entrenamiento():
    #1
    # prediccion = [1,1,1,0,2,2,0,0,0]
    #2{}
    prediccion = [100000]

    # Se obtiene el dataset retornado
    dataset = leerDataset()
 
    print("==== Inicia iteraciones de entrenamiento ====")
    modelo = implementacionKeras()

    # Compilar el algoritmo obteniendo la perdida cuadratica y realiza la optimizacion de que tanto aprendera el algoritmo
    modelo.compile(
        optimizer = keras.optimizers.Adam(0.00001),
        loss='mean_squared_error'
    )

    # Se obtiene el historial del aprendizaje con base a las epocas o numero de iteraciones

    historial = modelo.fit(dataset[0], dataset[1], epochs=4000)
    print(historial)

    # Se manda a llamar el interfaz que demostrara la magnitud de perdida y las epocas
    resultados(historial)
    modelo.save('./modelo/modelo.h5')
    modelo.save_weights('./modelo/pesos.h5')
    # Demostracion de una prediccion con base a un numero
    print('Prediccion en base a un numero: ', prediccion)
    resultado = modelo.predict([prediccion])
    print('Resultado: ', resultado)

def implementacionKeras():

    # Red neuronal Keras
    # Funsion de activacion
    capa = keras.layers.Dense(units= 200, input_shape=[1], activation='linear')
    capa1 = keras.layers.Dense(units= 200, activation='linear')
    capa2 = keras.layers.Dense(units= 200, activation='linear')
    capa3 = keras.layers.Dense(units= 200, activation='linear')
    # capa4 = keras.layers.Dense(units= 100, activation='sigmoid')
    # capa2 = keras.layers.Dense(units= 100, activation='linear')
    # capa3 = keras.layers.Dense(units= 100, activation='linear')
    capa4 = keras.layers.Dense(units= 1, activation='linear')
    # Modelo de la grafica a utilizar
    modelo = keras.Sequential([capa, capa1, capa2, capa3, capa4])

    return modelo

def resultados(historial):
    plt.xlabel('Numero de epoca')
    plt.ylabel('Magnitud de perdida')
    plt.plot(historial.history['loss'])
 
    plt.show()

entrenamiento()
# leerDataset()
