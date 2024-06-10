import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
import numpy as np
import plotly.express as px
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from timeit import default_timer as timer

class EmbeddedMethods:
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
    
    def split_data(self, data: pd.DataFrame, target_col: str) -> tuple:
        X = data.drop(['id', target_col], axis=1)
        Y = data[target_col]

        X_train, y_train, X_test, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        return X, Y, X_train, X_test, y_train, y_test

    def lasso_method(self, X, Y, X_train, y_train, X_test, y_test):
        start = timer()
        model = LassoCV(cv=5, random_state=0, max_iter=10000)
        model.fit(X_train, y_train)
        print(f'Best value for lambda is: {model.alpha_}')

        las_model = SelectFromModel(estimator = Lasso(alpha = model.alpha_, max_iter=10000), max_features = 30).fit(X, Y)
        print(las_model.get_feature_names_out())

        X_train_selected = las_model.transform(X_train)
        X_test_selected = las_model.transform(X_test)

        self.rfc.fit(X_train_selected, y_train)
        preds_rfc = self.rfc.predict(X_test_selected)
 
        accuracy_rfc = accuracy_score(y_test, preds_rfc)
        precision_rfc = precision_score(y_test, preds_rfc)
        recall_rfc = recall_score(y_test, preds_rfc)
        f1score_rfc = f1_score(y_test, preds_rfc ,average='weighted')
        stop = timer()
        print(f"Lasso selection took {stop-start} s.")
        print(f"Metrics with selected features:\n Accuracy: {accuracy_rfc}\n Precision: {precision_rfc}\n Recall: {recall_rfc}\n F1Score: {f1score_rfc}")

    def basic_tree_based_method(self, X_train, y_train, X_test, y_test) -> None:

        forest_fit = self.rfc.fit(X_train, y_train)
        preds_rfc = self.rfc.predict(X_test)

        accuracy_rfc = accuracy_score(y_test, preds_rfc)
        precision_rfc = precision_score(y_test, preds_rfc)
        recall_rfc = recall_score(y_test, preds_rfc)
        f1score_rfc = f1_score(y_test, preds_rfc ,average='weighted')

        print(f"Metrics with all features\n Accuracy RFC: {accuracy_rfc}\n Precision RFC: {precision_rfc}\n Recall RFC: {recall_rfc}\n F1Score RFC: {f1score_rfc}\n")
        importances = forest_fit.feature_importances_

        std = np.std([tree.feature_importances_ for tree in forest_fit.estimators_], axis=0)

        forest_importances = pd.Series(importances, index=X_test.columns).sort_values(ascending=False)

        fig = px.bar(forest_importances, 
                 x=forest_importances.values, 
                 y=forest_importances.index, 
                 orientation='h', 
                 labels={'x': 'Importance', 'y': 'Features'},
                 title='Feature Importances from Random Forest using MDI (Mean Decrease Impurity)')

        fig.write_html("Random Forest Feature Importances MDI.html")

        return importances
    
    def feature_selection_tree_based_method(self, X_train, y_train, X_test, y_test, importances, threshold):
        start = timer()
        selected_features = X_train.columns[importances > threshold]
        metric = {}
 
        # Use only the selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        self.rfc.fit(X_train_selected, y_train)
        preds_rfc = self.rfc.predict(X_test_selected)
        stop = timer()
 
        # Calculate the accuracy of the model with selected features
        accuracy_rfc = accuracy_score(y_test, preds_rfc)
        precision_rfc = precision_score(y_test, preds_rfc)
        recall_rfc = recall_score(y_test, preds_rfc)
        f1score_rfc = f1_score(y_test, preds_rfc ,average='weighted')
        metric["RFC"] = {"accuracy" : accuracy_rfc, "precision" : precision_rfc, "recall" : recall_rfc, "F1Score" : f1score_rfc}

        print(f"Feature selection took: {stop-start}s.")
        print(f"Selected features: {selected_features}\n")
        print(f"Accuracy RFC: {accuracy_rfc}\n Precision RFC: {precision_rfc}\n Recall RFC: {recall_rfc}\n, F1Score RFC: {f1score_rfc}")
        return metric


instance = EmbeddedMethods()
data = instance.prepare_data("/Users/patrykjaworski/Documents/Projekty/Feature-Selection/TESTY/Old/data.csv")
X, Y, X_train, y_train, X_test, y_test = instance.split_data(data, 'diagnosis')
importances = instance.basic_tree_based_method(X_train, y_train, X_test, y_test)
metric = instance.feature_selection_tree_based_method(X_train, y_train, X_test, y_test, importances, 0.009)
#instance.lasso_method(X, Y, X_train, y_train, X_test, y_test)








