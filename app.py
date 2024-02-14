# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 20:32:55 2024

@author: Usuario
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

def gastos_categoria(id_usuario):

    server = 'nelson-server.database.windows.net'
    database = 'nelson'
    username = 'administrador'
    password = 'Unix456nel!!'
     
     
    conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
     
    #id_usuario = 8
     # Establecer la conexión
    conn = pyodbc.connect(conn_str)
     
     # Consulta SQL para obtener los gastos agrupados por mes
    query = f"""
         select YEAR(cg.fechaRegistro) AS year,
                 MONTH(cg.fechaRegistro) AS month,
                 SUM(cg.monto) AS total_monto,
         		cgg.nombre as categoria
         from cuenta_gasto cg
         inner join categoria_gasto cgg on cgg.id = cg.categoriaGasto_id
         inner join cuenta c on c.id = cg.cuenta_id 
         inner join person p on p.id = c.person_id 
         where p.id = { id_usuario }
         GROUP BY YEAR(cg.fechaRegistro), MONTH(cg.fechaRegistro), cgg.nombre
     """
    print(query)
     # Cargar los datos en un DataFrame
    df = pd.read_sql_query(query, conn)
    
    print(df)
     # Preparación de los datos
    df = pd.get_dummies(df, columns=['categoria'], prefix='categoria', drop_first=False)
    print(df)
     
    query_categorias = """
         SELECT DISTINCT nombre as categoria
         FROM categoria_gasto
     """
    
     # Ejecuta la consulta y obtén las categorías en un DataFrame
    categorias_df = pd.read_sql_query(query_categorias, conn)
     
    
     
    print(categorias_df)
    
    # Obtén la lista completa de categorías desde tu base de datos o desde otro origen
    categorias_completas = categorias_df['categoria'].apply(lambda x: f'categoria_{x}').tolist()
     
     # Verifica las columnas presentes en tu DataFrame actual
    columnas_presentes = df.columns.tolist()
     
     # Agrega columnas faltantes con valores de 0 si no están presentes
    for categoria in categorias_completas:
         if categoria not in columnas_presentes:
             df[categoria] = 0
    print(df)
     
     # Dividir los datos en conjunto de entrenamiento y prueba (opcional)
     # Si deseas realizar un entrenamiento separado por mes, puedes omitir esta división
    X = df.drop(['total_monto'], axis=1)  # Características
    y = df['total_monto']  # Variable objetivo
     
     # Entrenamiento del modelo de regresión lineal (opcional si haces el entrenamiento por mes)
    model = LinearRegression()
     #model.fit(X_train, y_train)
    model.fit(X, y)
     
     
    mes_actual = datetime.now().month
    mes_siguiente = (mes_actual % 12) + 1
     
     # Crear una lista vacía para almacenar los datos obtenidos de la base de datos
    nuevos_datos = []
     
    print(mes_siguiente)
    
    data = [
        
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 1,
             'categoria_Alcohol': 0,
             'categoria_Celular': 0,
             'categoria_Cigarros': 0,
             'categoria_Cine': 0,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 0,
             'categoria_Indriver': 0,
             'categoria_Netflix': 0,
             'categoria_Restaurante': 0,
             'categoria_Sedalib': 0,
             'categoria_Spotify': 0
         },
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 0,
             'categoria_Alcohol': 1,
             'categoria_Celular': 0,
             'categoria_Cigarros': 0,
             'categoria_Cine': 0,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 0,
             'categoria_Indriver': 0,
             'categoria_Netflix': 0,
             'categoria_Restaurante': 0,
             'categoria_Sedalib': 0,
             'categoria_Spotify': 0
         },
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 0,
             'categoria_Alcohol': 0,
             'categoria_Celular': 1,
             'categoria_Cigarros': 0,
             'categoria_Cine': 0,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 0,
             'categoria_Indriver': 0,
             'categoria_Netflix': 0,
             'categoria_Restaurante': 0,
             'categoria_Sedalib': 0,
             'categoria_Spotify': 0
         },
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 0,
             'categoria_Alcohol': 0,
             'categoria_Celular': 0,
             'categoria_Cigarros': 1,
             'categoria_Cine': 0,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 0,
             'categoria_Indriver': 0,
             'categoria_Netflix': 0,
             'categoria_Restaurante': 0,
             'categoria_Sedalib': 0,
             'categoria_Spotify': 0
         },
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 0,
             'categoria_Alcohol': 0,
             'categoria_Celular': 0,
             'categoria_Cigarros': 0,
             'categoria_Cine': 1,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 0,
             'categoria_Indriver': 0,
             'categoria_Netflix': 0,
             'categoria_Restaurante': 0,
             'categoria_Sedalib': 0,
             'categoria_Spotify': 0
         },
         {
              'year': 2024,
              'month': mes_siguiente,
              'categoria_Adobe Suite': 0,
              'categoria_Alcohol': 0,
              'categoria_Celular': 0,
              'categoria_Cigarros': 0,
              'categoria_Cine': 0,
              'categoria_Golosinas': 1,
              'categoria_Hidrandina': 0,
              'categoria_Indriver': 0,
              'categoria_Netflix': 0,
              'categoria_Restaurante': 0,
              'categoria_Sedalib': 0,
              'categoria_Spotify': 0
         },
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 0,
             'categoria_Alcohol': 0,
             'categoria_Celular': 0,
             'categoria_Cigarros': 0,
             'categoria_Cine': 0,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 1,
             'categoria_Indriver': 0,
             'categoria_Netflix': 0,
             'categoria_Restaurante': 0,
             'categoria_Sedalib': 0,
             'categoria_Spotify': 0 
         },
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 0,
             'categoria_Alcohol': 0,
             'categoria_Celular': 0,
             'categoria_Cigarros': 0,
             'categoria_Cine': 0,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 0,
             'categoria_Indriver': 1,
             'categoria_Netflix': 0,
             'categoria_Restaurante': 0,
             'categoria_Sedalib': 0,
             'categoria_Spotify': 0 
         },
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 0,
             'categoria_Alcohol': 0,
             'categoria_Celular': 0,
             'categoria_Cigarros': 0,
             'categoria_Cine': 0,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 0,
             'categoria_Indriver': 0,
             'categoria_Netflix': 1,
             'categoria_Restaurante': 0,
             'categoria_Sedalib': 0,
             'categoria_Spotify': 0 
         },
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 0,
             'categoria_Alcohol': 0,
             'categoria_Celular': 0,
             'categoria_Cigarros': 0,
             'categoria_Cine': 0,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 0,
             'categoria_Indriver': 0,
             'categoria_Netflix': 0,
             'categoria_Restaurante': 1,
             'categoria_Sedalib': 0,
             'categoria_Spotify': 0 
         },
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 0,
             'categoria_Alcohol': 0,
             'categoria_Celular': 0,
             'categoria_Cigarros': 0,
             'categoria_Cine': 0,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 0,
             'categoria_Indriver': 0,
             'categoria_Netflix': 0,
             'categoria_Restaurante': 0,
             'categoria_Sedalib': 1,
             'categoria_Spotify': 0 
         },
         {
             'year': 2024,
             'month': mes_siguiente,
             'categoria_Adobe Suite': 0,
             'categoria_Alcohol': 0,
             'categoria_Celular': 0,
             'categoria_Cigarros': 0,
             'categoria_Cine': 0,
             'categoria_Golosinas': 0,
             'categoria_Hidrandina': 0,
             'categoria_Indriver': 0,
             'categoria_Netflix': 0,
             'categoria_Restaurante': 0,
             'categoria_Sedalib': 0,
             'categoria_Spotify': 1 
         }
     ]
    
    
    
    X_future = pd.DataFrame(data)
    print(X_future)
    # Obtén las columnas en el mismo orden que se usaron durante el entrenamiento
    columnas_ordenadas = X.columns.tolist()
    
    # Reorganiza las columnas de X_future para que coincidan con el orden
    X_future = X_future[columnas_ordenadas]
    
    
     # Realizar predicciones para el próximo mes (opcional si haces el entrenamiento por mes)
    y_future = model.predict(X_future).tolist()
    
    print(y_future)
    
    
    query2 = f"""
        SELECT YEAR(cg.fechaRegistro) AS year,
           MONTH(cg.fechaRegistro) AS month,
           SUM(cg.monto) AS total_monto,
           cgg.nombre AS categoria
    FROM cuenta_gasto cg
    INNER JOIN categoria_gasto cgg ON cgg.id = cg.categoriaGasto_id
    INNER JOIN cuenta c ON c.id = cg.cuenta_id 
    INNER JOIN person p ON p.id = c.person_id 
    WHERE p.id = { id_usuario }
          AND YEAR(cg.fechaRegistro) = YEAR(GETDATE())
          AND MONTH(cg.fechaRegistro) = MONTH(GETDATE())
    GROUP BY YEAR(cg.fechaRegistro), MONTH(cg.fechaRegistro), cgg.nombre;
    
    """
    mesActual = pd.read_sql_query(query2, conn)
    # Cierra la conexión
    conn.close()
   
    return  categorias_completas, y_future, mesActual

def gastos_mes(id_usuario):
    
   server = 'nelson-server.database.windows.net'
   database = 'nelson'
   username = 'administrador'
   password = 'Unix456nel!!'
    
    
   conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    
   conn = pyodbc.connect(conn_str)
    
    # Consulta SQL para obtener los gastos agrupados por mes
   query = f"""
           SELECT 
           	YEAR(CONVERT(date, cg.fechaRegistro)) AS Anio,
               MONTH(CONVERT(date, cg.fechaRegistro)) AS Mes,
           	SUM(cg.monto) AS TotalMonto
           FROM cuenta_gasto cg
           INNER JOIN cuenta c ON c.id = cg.cuenta_id 
           INNER JOIN person p ON p.id = c.person_id 
           WHERE p.id = 8
           GROUP BY 
           YEAR(CONVERT(date, cg.fechaRegistro)),
           MONTH(CONVERT(date, cg.fechaRegistro))
           ORDER BY  YEAR(CONVERT(date, cg.fechaRegistro)),
           MONTH(CONVERT(date, cg.fechaRegistro))
    """
   cursor = conn.cursor()
   cursor.execute(query)
      
   # Obtener los resultados de la consulta y almacenarlos en un array para TotalMonto
   resultados = cursor.fetchall()
   TotalMonto = np.array([r.TotalMonto for r in resultados])
      
   # Generar un array con 30 días para cada mes
   dias_por_mes = np.full(len(resultados), 30)
      
   # Cerrar la conexión
   conn.close()
      
   # Verificar los resultados
   print("TotalMonto:", TotalMonto)
   print("dias_por_mes:", dias_por_mes)


   # Datos de gastos por mes (en dólares)
   gastos_por_mes = TotalMonto
   dias_por_mes = dias_por_mes

   # Crear un array con los días del mes
   dias_del_mes = np.arange(1, len(gastos_por_mes) + 1).reshape(-1, 1)

   # Asegurarse de que el número de días en el mes para entrenamiento coincida con el número de días para predicción
   if len(dias_del_mes) != len(dias_por_mes):
       raise ValueError("El número de días en el mes para entrenamiento y predicción no coincide.")

   # Entrenar el modelo de regresión lineal
   modelo = LinearRegression()
   modelo.fit(dias_del_mes, gastos_por_mes)

   # Hacer la predicción para el siguiente mes
   siguiente_dia = len(dias_por_mes) + 1
   prediccion = modelo.predict([[siguiente_dia]])

   prediccion_gastos_siguiente_mes = prediccion[0]
   return prediccion_gastos_siguiente_mes

def gastos_hormida(id_usuario):
    # Generar transacciones simuladas
    np.random.seed(0)
    
    # Simulación de 80 transacciones
    n_transactions = 80
    transacciones = pd.DataFrame({
        'monto': np.random.randint(1, 100, n_transactions),
        'fechaRegistro': pd.date_range(start='2024-01-01', periods=n_transactions, freq='D'),
        'cuenta_id': np.random.randint(1, 5, n_transactions),
        'categoriaGasto': np.random.choice(['Alimentos', 'Transporte', 'Entretenimiento', 'Ropa'], n_transactions),
        'tipoGasto': np.random.choice(['Tarjeta', 'Efectivo'], n_transactions)
    })
    
    # Agregar una columna para indicar si es un gasto hormiga (por defecto, todos son False)
    transacciones['gasto_hormiga'] = False
    
    # Supongamos que los gastos menores a $30 se consideran gastos hormiga
    transacciones.loc[transacciones['monto'] < 30, 'gasto_hormiga'] = True
    
    # Definir características relevantes para detectar gastos hormiga
    features = ['monto', 'cuenta_id']
    
    # Dividir el conjunto de datos en entrenamiento y prueba
    X = transacciones[features]
    y = transacciones['gasto_hormiga']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar un modelo de clasificación
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular la precisión del modelo
    #accuracy = accuracy_score(y_test, y_pred)
    #print("Precisión del modelo:", accuracy)
    
    # Identificar las categorías de gastos más comunes entre las transacciones clasificadas como gastos hormiga
    categorias_gastos_hormiga = transacciones.loc[(transacciones['gasto_hormiga'] == True), 'categoriaGasto'].value_counts()
    categorias_gastos_hormiga_lista = categorias_gastos_hormiga.index.tolist()

    # Mostrar la lista de categorías de gastos hormiga
    print("Categorías de gastos hormiga como lista:")
    print(categorias_gastos_hormiga_lista)
    return categorias_gastos_hormiga_lista

@app.route('/gastosCategoria/<int:id_usuario>', methods=['GET'])
def modelo_categoria(id_usuario):
    # Llamar a la función para cargar datos y entrenar el modelo
    categorias, datos, datosActuales = gastos_categoria(id_usuario)
    
    # Crear un solo diccionario con las categorías y predicciones
    # resultado = {categoria: dato for categoria, dato in zip(categorias, datos)}
    datosActuales = datosActuales[["categoria", "total_monto"]]
    
    resultado = {
        "actual": datosActuales.to_dict(orient='records'),  # Convertir DataFrame a lista de diccionarios
        "futuro": [{categoria: dato for categoria, dato in zip(categorias, datos)}]
    }
    
    return jsonify(resultado)

@app.route('/gastosMes/<int:id_usuario>', methods=['GET'])
def modelo_mes(id_usuario):
       # Llamar a la función para cargar datos y entrenar el modelo
       prediccion_gastos_siguiente_mes = gastos_mes(id_usuario)
       return jsonify({"futuro": prediccion_gastos_siguiente_mes})

@app.route('/gastosHormiga/<int:id_usuario>', methods=['GET'])
def modelo_hormiga(id_usuario):
       # Llamar a la función para cargar datos y entrenar el modelo
      categorias_gastos_hormiga_lista = gastos_hormida(id_usuario)
      return jsonify({"gastos_hormida": categorias_gastos_hormiga_lista} )


if __name__ == '__main__':
    app.run(debug=True)