def create_txt(input_text, output_path):
    """
    Crea un archivo .txt a partir del texto dado
    """
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(input_text)

def extract_text_from_txt(txt_path):
    """
    Extrae el texto de un archivo .txt
    """
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()
