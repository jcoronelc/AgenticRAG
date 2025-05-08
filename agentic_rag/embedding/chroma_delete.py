import chromadb

# Crear cliente

persist_directory = "../data/output/chroma/persistent_directory"
    
      
client_bd = chromadb.PersistentClient(path=persist_directory)
collection_name =  ["bdvt1", "bdvt2", "bdvt3", "bdvt4", "bdvt5", "bdvt6", "bdvt7", "bdvt8", "bdvt9", "bdvt10" ]
for name in collection_name:
    try:
        collection = client_bd.get_collection(name)
        print(f"Usando colección existente: '{name}'")
        client_bd.delete_collection(name=name)

        print(f"Colección '{name}' eliminada exitosamente.")
    except Exception as e:
        print("Ya existe")



