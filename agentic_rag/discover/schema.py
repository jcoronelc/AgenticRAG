import pandas as pd

class Schema:
    def __init__(self, df: pd.DataFrame, sample_size: int):
        self.df = df.sample(n=sample_size, random_state=42)

    def get_schema(self):
        schema = {col: str(self.df[col].dtype) for col in self.df.columns}
        return schema, self.df
    
