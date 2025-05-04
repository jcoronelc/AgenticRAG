import pandas as pd

class Process:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def process_data(self):
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        text_cols = self.df.select_dtypes(include=['object']).columns
        for col in numeric_cols:
            self.df[col] = self.df[col].apply(lambda x: self.clean_value(x, is_numeric=True))
        for col in text_cols:
            self.df[col] = self.df[col].apply(lambda x: self.clean_value(x, is_numeric=False))
            
        self.df.columns = self.df.columns.str.lower()
        #self.df.to_csv('./data/output/data_procesada.csv')
        return self.df

    def clean_value(self, value, is_numeric):
        if pd.isna(value) or value == '':
            return 0 if is_numeric else "S/N"
        if is_numeric:
            return float(str(value).replace(',', '.'))  # Convertir a numero
        return str(value).lower().strip()  # Convertir a texto y normalizar
