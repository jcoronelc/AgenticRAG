from jinja2 import Template
import pandas as pd

class DocumentGeneration:
    def __init__(self, template: Template, formatted_df: pd.DataFrame):
        self.template = template
        self.formatted_df = formatted_df

    def generate_documents(self):
        documents = []
        for _, row in self.formatted_df.iterrows():
            rendered_doc = self.template.render(**row.to_dict())
            documents.append(rendered_doc)
        return documents
