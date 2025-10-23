import pandas as pd

# Definir los nombres de los archivos
file_clientes = 'clientes.csv'
file_productos = 'productos.csv'
file_pedidos = 'pedidos.csv'

# --- Tarea de Extracción ---
# Cargar los CSVs en DataFrames
df_clientes = pd.read_csv(file_clientes)
df_pedidos = pd.read_csv(file_pedidos)
df_productos = pd.read_csv(file_productos)

# --- Verificación Inicial ---
# Siempre es bueno ver que se cargó bien
print("--- Clientes ---")
print(df_clientes.head())
print("\n--- Pedidos ---")
print(df_pedidos.head())
print("\n--- Productos ---")
print(df_productos.head())

# Y revisar los tipos de datos
print("\n--- Tipos de Datos (Info) ---")
print("Clientes Info:")
df_clientes.info()
print("\nPedidos Info:")
df_pedidos.info()
print("\nProductos Info:")
df_productos.info()
# --- Tarea de Transformación (Limpieza) ---

print("\n--- Iniciando Limpieza y Transformación ---")

# 1. Arreglar df_productos: Convertir 'Precio_Unitario' a numérico
#    Quitamos el '$' y convertimos a float (número decimal)
print("Limpiando 'Precio_Unitario'...")
# Usamos .replace() para quitar el '$' y los espacios (por si acaso)
df_productos['Precio_Unitario'] = df_productos['Precio_Unitario'].str.replace('$', '').str.strip()
# Convertimos la columna a tipo numérico (float)
df_productos['Precio_Unitario'] = pd.to_numeric(df_productos['Precio_Unitario'])


# 2. Arreglar df_pedidos: Manejar valores nulos
#    Tenemos un 'ID_Producto' nulo. Ese pedido no nos sirve.
print("Manejando nulos en 'ID_Producto'...")
# Vemos cuántos nulos hay antes
print(f"Nulos antes: {df_pedidos['ID_Producto'].isnull().sum()}")
# Eliminamos las filas donde 'ID_Producto' sea nulo
df_pedidos.dropna(subset=['ID_Producto'], inplace=True)
# Vemos cuántos nulos hay después (debería ser 0)
print(f"Nulos después: {df_pedidos['ID_Producto'].isnull().sum()}")

# Ahora que no hay nulos, podemos convertir 'ID_Producto' a entero (int)
df_pedidos['ID_Producto'] = df_pedidos['ID_Producto'].astype(int)


# 3. Arreglar Fechas: Convertir columnas de fecha a datetime
print("Convirtiendo columnas de fecha...")
df_clientes['Fecha_Registro'] = pd.to_datetime(df_clientes['Fecha_Registro'])
df_pedidos['Fecha_Pedido'] = pd.to_datetime(df_pedidos['Fecha_Pedido'])

# --- Verificación de Limpieza ---
print("\n--- Verificación de Datos Limpios ---")
print("\nProductos Info (limpio):")
df_productos.info()
print("\nPedidos Info (limpio):")
df_pedidos.info()
print("\nClientes Info (limpio):")
df_clientes.info()
# --- 2.2 TRANSFORMACIÓN (Combinación y Feature Engineering) ---

print("\n--- Combinando (Merging) DataFrames ---")

# Paso 1: Unir pedidos con productos (para saber el precio y categoría de cada pedido)
# Usamos un 'left' merge para mantener todos nuestros pedidos limpios
# La llave de unión es 'ID_Producto'
df_pedidos_productos = pd.merge(
    df_pedidos, 
    df_productos, 
    on='ID_Producto', 
    how='left'
)

# Paso 2: Feature Engineering - Calcular el ingreso por línea de pedido
# Multiplicamos la cantidad por el precio ya limpio
df_pedidos_productos['Ingreso_Linea'] = df_pedidos_productos['Cantidad'] * df_pedidos_productos['Precio_Unitario']

# Paso 3: Unir la tabla combinada con los clientes (para saber quién compró)
# La llave de unión es 'ID_Cliente'
df_maestro = pd.merge(
    df_pedidos_productos,
    df_clientes,
    on='ID_Cliente',
    how='left'
)

# --- Verificación de la Tabla Maestra ---
print("\n--- Tabla Maestra Combinada (head) ---")
# Mostramos las columnas clave para ver si la unión fue exitosa
print(df_maestro[['ID_Pedido', 'ID_Cliente', 'Nombre', 'Nombre_Producto', 'Cantidad', 'Precio_Unitario', 'Ingreso_Linea', 'Categoria']].head())

print("\n--- Tabla Maestra Combinada (info) ---")
df_maestro.info()
# --- 2.3 TRANSFORMACIÓN (Agregación) ---
# Ahora tenemos la tabla maestra, pero está a nivel de transacción (una fila por producto comprado).
# Queremos agregarla a nivel de cliente (una fila por cliente).

print("\n--- Agregando datos por cliente ---")

# Esta es la parte difícil: calcular la categoría favorita.
# 1. Agrupamos por cliente y categoría, y contamos cuántos productos compró de cada una.
# 2. Luego, encontramos el índice (la categoría) del valor máximo para cada cliente.
categoria_favorita = df_maestro.groupby('ID_Cliente')['Categoria'].apply(
    lambda x: x.mode().iloc[0] # .mode() encuentra el valor más frecuente
).rename('Categoria_Favorita')


# Ahora definimos las agregaciones principales.
# Usaremos .agg() para hacer varios cálculos a la vez.
agregaciones = {
    'Ingreso_Linea': 'sum',    # Suma total de lo que gastó
    'ID_Pedido': 'nunique',  # Conteo de pedidos únicos
}

# 1. Agrupamos por 'ID_Cliente' y 'Nombre' (para mantener el nombre)
# 2. Aplicamos las agregaciones
df_analitico_clientes = df_maestro.groupby(
    ['ID_Cliente', 'Nombre']
).agg(agregaciones).reset_index()

# Renombramos las columnas para que sean claras
df_analitico_clientes = df_analitico_clientes.rename(columns={
    'Ingreso_Linea': 'Gasto_Total',
    'ID_Pedido': 'Pedidos_Totales'
})

# 3. Finalmente, unimos nuestro reporte analítico con la categoría favorita
df_analitico_clientes = pd.merge(
    df_analitico_clientes,
    categoria_favorita,
    on='ID_Cliente',
    how='left'
)


# --- Verificación de la Tabla Analítica ---
print("\n--- TABLA ANALÍTICA FINAL (por cliente) ---")
print(df_analitico_clientes)


# --- 3. LOAD (Carga) ---
# ¡El paso final! Guardamos nuestro resultado en un nuevo archivo.
# Usar Parquet es mucho más eficiente que CSV.
output_file = 'reporte_analitico_clientes.parquet'
print(f"\n--- Guardando reporte final en: {output_file} ---")

df_analitico_clientes.to_parquet(output_file, index=False)

print("\n¡Pipeline ETL completado exitosamente! ✨")