import chromadb

# Crear cliente

persist_directory = "./data/output/chroma/persistent_directory"
    
      
client_bd = chromadb.PersistentClient(path=persist_directory)
collection_name = "bd1" 
collection = client_bd.get_collection(collection_name)


print(f"Usando colección existente: '{collection_name}'")

# Nombre de la colección que quieres borrar


# Eliminar la colección
client_bd.delete_collection(name=collection_name)

print(f"Colección '{collection_name}' eliminada exitosamente.")
