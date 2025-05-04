import pandas as pd

class FormattingText:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def format_data(self):
        for col in self.df.select_dtypes(include=['datetime']):
            self.df[col] = self.df[col].dt.strftime('%Y-%m-%d')
        for col in self.df.select_dtypes(include=['float', 'int']):
            self.df[col] = self.df[col].round(2)
        return self.df
