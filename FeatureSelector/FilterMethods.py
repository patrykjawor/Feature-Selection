import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from timeit import default_timer as timer

class FilterMethods:
    def __init__(self) -> None:
        self.gbc = GradientBoostingClassifier()
        self.rfc = RandomForestClassifier()

    def prepare_data(self, filepath : str) -> pd.DataFrame:
        data = pd.read_csv(filepath_or_buffer=filepath)
        label_encoder = preprocessing.LabelEncoder()
        nan_count = data.isnull().sum()
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
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        return X, Y, X_train, X_test, y_train, y_test
    
    def occurence_of_classes(self, data, target_col):
        class_counts = data[target_col].value_counts()
        return class_counts


    def basic_classification(self, X_train, Y_train, X_test, Y_test) -> dict:
        start = timer()
        self.gbc.fit(X=X_train, y=Y_train)
        self.rfc.fit(X=X_train, y=Y_train)
        preds_gbc = self.gbc.predict(X_test)
        preds_rfc = self.rfc.predict(X_test)
        stop = timer()

        metrics = {}

        accuracy_gbc = round(accuracy_score(Y_test, preds_gbc), 3)
        precision_gbc = round(precision_score(Y_test, preds_gbc), 3)
        recall_gbc = round(recall_score(Y_test, preds_gbc), 3)
        f1_gbc = round(f1_score(Y_test, preds_gbc, average='weighted'), 3)

        metrics['GradientBoost'] = {'features': X_train.columns, 
                                    'accuracy': accuracy_gbc, 
                                    'precision': precision_gbc, 
                                    'recall': recall_gbc, 
                                    'f1_score': f1_gbc}
        
        print(f"Gradient boost Classifier: accuracy: {accuracy_gbc}, precision: {precision_gbc}, recall: {recall_gbc}, f1_score: {f1_gbc}")

        accuracy_rfc = round(accuracy_score(Y_test, preds_rfc), 3)
        precision_rfc = round(precision_score(Y_test, preds_rfc), 3)
        recall_rfc = round(recall_score(Y_test, preds_rfc), 3)
        f1_rfc = round(f1_score(Y_test, preds_rfc, average='weighted'), 3)

        metrics['RandomForest'] = {'features': X_train.columns, 
                                   'accuracy': accuracy_rfc, 
                                   'precision': precision_rfc, 
                                   'recall': recall_rfc, 'f1_score': f1_rfc}
        
        print(f"Random Forest Classifier: accuracy: {accuracy_rfc}, precision: {precision_rfc}, recall: {recall_rfc}, f1_score: {f1_rfc}")
        print(f"Basic classification took {stop-start} s.")

        return metrics

    def plot_variance(self, X, X_train, filename: str) -> None:
        scaler = MinMaxScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        x = X.columns
        y = scaled_X_train.var(axis=0)
        df = pd.DataFrame({'Feature': x, 'Variance': y})
        df = df.sort_values(by="Variance", ascending=False)
        fig = px.bar(df, x="Feature", y="Variance", color="Feature", title="Variance of Features")
        fig.write_html(filename)

    def variance_threshold_method(self, X_train, Y_train, X_test, Y_test, threshold: float) -> dict:
        start = timer()
        scaler = MinMaxScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        X_train_copy = X_train.copy()
        X_train_copy[:] = scaled_X_train
        var_threshold = VarianceThreshold(threshold=threshold)
        var_threshold.fit(X_train_copy)
        # Get the indices of non-constant columns
        constant_columns_indices = [i for i, var in enumerate(var_threshold.get_support()) if not var]
        # Get the names of constant columns
        constant_columns = X_train_copy.columns[constant_columns_indices]
        # Print or use the constant column names
        print(constant_columns)
        X_train = X_train.drop(columns = constant_columns)
        X_test = X_test.drop(columns = constant_columns)

        self.gbc.fit(X=X_train, y=Y_train)
        self.rfc.fit(X=X_train, y=Y_train)
        preds_gbc = self.gbc.predict(X_test)
        preds_rfc = self.rfc.predict(X_test)

        stop = timer()

        metrics = {}

        accuracy_gbc = round(accuracy_score(Y_test, preds_gbc), 3)
        precision_gbc = round(precision_score(Y_test, preds_gbc), 3)
        recall_gbc = round(recall_score(Y_test, preds_gbc), 3)
        f1_gbc = round(f1_score(Y_test, preds_gbc, average='weighted'), 3)
        metrics['GradientBoost'] = {'features': X_train.columns, 'accuracy': accuracy_gbc,
                                    'precision': precision_gbc, 'recall': recall_gbc, 
                                    'f1_score': f1_gbc}
        
        print(f"Gradient boost Classifier: accuracy: {accuracy_gbc}, precision: {precision_gbc}, recall: {recall_gbc}, f1_score: {f1_gbc}")

        accuracy_rfc = round(accuracy_score(Y_test, preds_rfc), 3)
        precision_rfc = round(precision_score(Y_test, preds_rfc), 3)
        recall_rfc = round(recall_score(Y_test, preds_rfc), 3)
        f1_rfc = round(f1_score(Y_test, preds_rfc, average='weighted'), 3)
        
        metrics['RandomForest'] = {'features':X_train.columns, 'accuracy': accuracy_rfc, 
                                   'precision': precision_rfc, 'recall': recall_rfc, 
                                   'f1_score': f1_rfc}
        
        print(f"Random Forest Classifier: accuracy: {accuracy_rfc}, precision: {precision_rfc}, recall: {recall_rfc}, f1_score: {f1_rfc}")
        print(f"Optimization took {stop - start} s.")

        return metrics
    
    def plot_correlation(self, X, filename: str) -> None:
        figsize = (20, 20)
        _, ax = plt.subplots(figsize=figsize)
        sns.heatmap(X.corr(), annot=True, linewidths=.6, fmt='.2f', ax=ax)
        plt.savefig(filename)

    def correlation_method(self, data, X_train, Y_train, X_test, Y_test, threshold : float) -> dict:
        highly_correlated = set()
        corr_thresh = threshold
        corr_matrix = data.corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i,j]) > corr_thresh:
                    colname = corr_matrix.columns[i]
                    highly_correlated.add(colname)

        print("HIGHLY CORRELATED:", highly_correlated)

        X_train = X_train.drop(columns=highly_correlated)
        X_test = X_test.drop(columns=highly_correlated)

        self.gbc.fit(X=X_train, y=Y_train)
        self.rfc.fit(X=X_train, y=Y_train)
        preds_gbc = self.gbc.predict(X_test)
        preds_rfc = self.rfc.predict(X_test)

        metrics = {}

        accuracy_gbc = round(accuracy_score(Y_test, preds_gbc), 3)
        precision_gbc = round(precision_score(Y_test, preds_gbc), 3)
        recall_gbc = round(recall_score(Y_test, preds_gbc), 3)
        f1_gbc = round(f1_score(Y_test, preds_gbc, average='weighted'), 3)
        
        metrics['GradientBoost'] = {'accuracy': accuracy_gbc, 'precision': precision_gbc,
                                    'recall': recall_gbc, 'f1_score': f1_gbc}
        
        print(f"Gradient boost Classifier: accuracy: {accuracy_gbc}, precision: {precision_gbc}, recall: {recall_gbc}, f1_score: {f1_gbc}")

        accuracy_rfc = round(accuracy_score(Y_test, preds_rfc), 3)
        precision_rfc = round(precision_score(Y_test, preds_rfc), 3)
        recall_rfc = round(recall_score(Y_test, preds_rfc), 3)
        f1_rfc = round(f1_score(Y_test, preds_rfc, average='weighted'), 3)

        metrics['RandomForest'] = {'accuracy': accuracy_rfc, 'precision': precision_rfc,
                                   'recall': recall_rfc, 'f1_score': f1_rfc}
        
        print(f"Random Forest Classifier: accuracy: {accuracy_rfc}, precision: {precision_rfc}, recall: {recall_rfc}, f1_score: {f1_rfc}")

        return metrics

    def mutual_info_method(self, X_train, Y_train, X_test, Y_test) -> None:
        mutual_info = mutual_info_classif(X_train, Y_train)
        mutual_info = pd.Series(mutual_info)
        mutual_info.index = X_train.columns
        mutual_info.sort_values(ascending=False).plot.bar()

        accuracy_list = []
        recall_list = []
        precision_list = []
        f1_score_list = []
        feature_names = []

        start = timer()
        for k in range (1 , len(X_train.columns) + 1):
            selection = SelectKBest(mutual_info_classif, k=k)
            selection.fit(X_train, Y_train)
            sel_X_train_v3 = selection.transform(X_train)
            sel_X_test_v3 = selection.transform(X_test)

            selected_features = X_train.columns[selection.get_support()]
            feature_names.append(selected_features.tolist())

            self.gbc.fit(sel_X_train_v3, Y_train)
            kbest_preds = self.gbc.predict(sel_X_test_v3)
            accuracy_gbc_kbest = round(accuracy_score(Y_test, kbest_preds), 3)
            precision_gbc_kbest = round(precision_score(Y_test, kbest_preds), 3)
            recall_gbc_kbest = round(recall_score(Y_test, kbest_preds), 3)
            f1_score_kbest = round(f1_score(Y_test, kbest_preds, average='weighted'), 3)
            accuracy_list.append(accuracy_gbc_kbest)
            precision_list.append(precision_gbc_kbest)
            recall_list.append(recall_gbc_kbest)
            f1_score_list.append(f1_score_kbest)
        stop = timer()

        f1_score_list = pd.Series(f1_score_list)
        accuracy_list = pd.Series(accuracy_list)
        precision_list = pd.Series(precision_list)
        recall_list = pd.Series(recall_list)

        df_f1 = pd.DataFrame({'Feature_Num': range(1, len(X_train.columns) + 1), 'F1_Score': f1_score_list})
        df_f1['Feature_Num'] = df_f1['Feature_Num'].astype(int)
        df_f1 = df_f1.sort_values(by="F1_Score", ascending=True)

        df_accuracy = pd.DataFrame({'Feature_Num': range(1, len(X_train.columns) + 1), 'Accuracy': accuracy_list})
        df_accuracy['Feature_Num'] = df_accuracy['Feature_Num'].astype(int)
        df_accuracy = df_accuracy.sort_values(by="Accuracy", ascending=True)

        df_precision = pd.DataFrame({'Feature_Num': range(1, len(X_train.columns) + 1), 'Precision': precision_list})
        df_precision['Feature_Num'] = df_precision['Feature_Num'].astype(int)
        df_precision = df_precision.sort_values(by="Precision", ascending=True)

        df_recall = pd.DataFrame({'Feature_Num': range(1, len(X_train.columns) + 1), 'Recall': precision_list, 'Features': feature_names})
        df_recall['Feature_Num'] = df_recall['Feature_Num'].astype(int)
        df_recall = df_recall.sort_values(by="Recall", ascending=True)

        # Merge DataFrames on 'Feature_Num'
        df_combined = pd.merge(df_f1, df_accuracy, on='Feature_Num')
        df_combined = pd.merge(df_combined, df_precision, on='Feature_Num')
        df_combined = pd.merge(df_combined, df_recall, on='Feature_Num')

        # Optional: sort by Feature_Num if needed
        df_combined = df_combined.sort_values(by="Feature_Num")
        df_combined.to_excel("mutual_info.xlsx")

        fig_f1 = px.bar(df_f1, x="Feature_Num", y="F1_Score", color="Feature_Num", title="F1 Scores vs Feature Number")
        fig_f1.write_html("f1_score_mutual_info.html")

        fig_accuracy = px.bar(df_accuracy, x="Feature_Num", y="Accuracy", color="Feature_Num", title="Accuracy vs Feature Number")
        fig_accuracy.write_html("accuracy_mutual_info.html")

        fig_precision = px.bar(df_precision, x="Feature_Num", y="Precision", color="Feature_Num", title="Precision vs Feature Number")
        fig_precision.write_html("precision_mutual_info.html")

        fig_recall = px.bar(df_recall, x="Feature_Num", y="Recall", color="Feature_Num", title="Recall vs Feature Number")
        fig_recall.write_html("recall_mutual_info.html")
        
        print(f"Optimization took {stop-start} s.")


instance = FilterMethods()
data = instance.prepare_data("/Users/patrykjaworski/Documents/Projekty/Feature-Selection/TESTY/Old/data.csv")
X, Y, X_train, X_test, y_train, y_test = instance.split_data(data, "diagnosis")
#print(f"{instance.occurence_of_classes(data, "diagnosis")}\n")
instance.plot_correlation(X, "correlation.png")
#instance.basic_classification(X_train, y_train, X_test, y_test)
instance.correlation_method(data, X_train, y_train, X_test, y_test, threshold=0.99)


#basic_classification_metrics = instance.basic_classification(X_train, y_train, X_test, y_test)
#instance.plot_variance(X, X_train, "variance_plot.html")
#var_classification_metrics = instance.variance_threshold_method(X_train, y_train, X_test, y_test, 0.02)
# print(f"Basic classification metrics: \n{basic_classification_metrics}\n")
# print(f"Variation threshold classification metrics:\n{var_classification_metrics}\n")
instance.mutual_info_method(X_train, y_train, X_test, y_test)
