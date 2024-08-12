import pandas as pd

class MyDataFrame(object):
    def __init__(self, data, columns):
        self.df = pd.DataFrame(data, columns=columns)

    def to_dict(self):
        return self.df.to_dict(orient='records')
    
    def add_row(self, row):
        addition = pd.DataFrame([row])
        self.df = pd.concat([self.df, addition], ignore_index=True)

    def remove_row(self, row):
        self.df = self.df.drop(row)

    @classmethod
    def from_csv(cls, file_path, columns):
        data = pd.read_csv(file_path)
        return cls(data.to_dict(orient='records'), columns)

    def save_csv(self, file_path):
        self.df.to_csv(file_path, index=False)