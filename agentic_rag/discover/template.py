
import re
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from jinja2 import Template
from nltk.corpus import stopwords

stopwords_es = set(stopwords.words('spanish'))


class TemplateGenerator:
    def __init__(self, schema: dict):
        self.schema = schema

    def create_template(self, format_type):
        """
        Genera una plantilla según el formato especificado
        Tipo de formato ("general" o "detailed")
        """
        
        if format_type == "general":
            template_str = ""
            for idx, col in enumerate(self.schema.keys()):
                template_str += f"{col}: {{{{{col}}}}}"
                if idx < len(self.schema) - 1:
                    template_str += ", "
        elif format_type == "detailed":
            
            template_str = (
                "En el periodo académico {{ periodo_academico }}, el estudiante con ID {{ identificador_estudiante }} "
                "con el nombre {{ nombres }} {{ apellidos }} y edad de {{ edad }} e identificandose como del género {{ genero }}) "
                "tiene la direccion de correo {{ correo }},"
                "está inscrito en el programa {{ programa_nombre }} de nivel {{ programa_nivel_nombre }} en la modalidad "
                "{{ programa_modalidad_nombre }}. Está cursando la asignatura {{ curso_nombre }} "
                "código: {{ curso_codigo }}, paralelo: {{ paralelo }}. Su sistema de evaluación es '{{ sistema_evaluacion }}' "
                "y obtuvo las siguientes notas: Bimestre 1: {{ bim1_nota }}, Bimestre 2: {{ bim2_nota }}. "
                "Estado de calificación: {{ estado_calificacion }}. Nota final: {{ final_nota }}, "
                "Nota total final: {{ totalfin_nota }}.\n\n"
            )
        else:
            raise ValueError("Invalid format_type. Use 'general' or 'detailed'.")


        return Template(template_str)
    
    def clean_text(self, text):
        """
        Limpia el texto eliminando caracteres especiales y stopwords
        """
        # conservar solo letras, numeros y espacios
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Eliminar stopwords
        # text = " ".join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])
        text = " ".join([word for word in text.split() if word.lower() not in stopwords_es])
        return text
