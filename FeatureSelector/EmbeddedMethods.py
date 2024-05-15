import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class EmebededMethods:
    def __init__(self) -> None:
        self.gbc = GradientBoostingClassifier()
        self.rfc = RandomForestClassifier()

    def prepare_data(self, filepath: str) -> pd.DataFrame:
        data = pd.read_csv(filepath_or_buffer=filepath)
        label_encoder = preprocessing.LabelEncoder()
        nan_count = data.isnull().sum()
        print(nan_count)
        data = (
            data
            .loc[:, ~data.columns.str.contains('^Unnamed')]  # Remove columns starting with 'Unnamed'
            .apply(lambda col: col.fillna("NO INFO") if is_object_dtype(col) else col)  # Fill missing values with "NO INFO" for object columns (str)
            .apply(lambda col: label_encoder.fit_transform(col) if is_object_dtype(col) else col)  # Label encode object columns
            .apply(lambda col: col.fillna(col.median()) if is_numeric_dtype(col) else col)  # Fill missing values with median for numeric columns
        )
        return data

    def lasso_method(self) -> None:
        pass

    def tree_based_method(self) -> None:
        pass






