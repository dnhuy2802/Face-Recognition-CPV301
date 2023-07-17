import pandas as pd
from ...utils.constants import MODEL_TABLE_PATH


class ModelTable:
    def __init__(self, table_path: str = MODEL_TABLE_PATH):
        self.table_path = table_path
        self.dataframe = pd.DataFrame(
            columns=['name', 'type', 'path', 'accuracy'])

    def load_table(self):
        self.dataframe = pd.read_csv(self.table_path)

    def save(self):
        self.dataframe.to_csv(self.table_path, index=False)

    def add_model(self, name: str, model_type: str, path: str, accuracy: float):
        self.dataframe.loc[len(self.dataframe)] = [
            name, model_type, path, accuracy]
        self.save()

    def delete_model(self, name: str):
        self.dataframe = self.dataframe[self.dataframe['name'] != name]
        self.save()

    def get_model(self, name: str, return_path: bool = False):
        model = self.dataframe[self.dataframe['name'] == name]
        if return_path:
            return model['path'].values[0]
        return model
