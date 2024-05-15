import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from numba import jit
from itertools import accumulate
import operator


class WrapperMethods:
    def __init__(self) -> None:
        self.gbc = GradientBoostingClassifier()
        self.rfc = RandomForestClassifier()
        self.rng = np.random.default_rng(seed=25)
        
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
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        return X, Y, X_train, X_test, y_train, y_test

    def forward_selection_method(self, X_train, X_test, Y_train, Y_test) -> tuple:
        best_features_rfc, best_features_gbc = set(), set()
        metrics = {}
        f1_score_gbc_best = 0
        f1_score_rfc_best = 0
        f1_gbc_list = []
        f1_rfc_list = []
        gbc_score_improvement = []
        rfc_score_improvement = []
        features_mask_gbc = np.zeros(X_train.shape[1], dtype=bool)
        features_mask_rfc = np.zeros(X_train.shape[1], dtype=bool)
        num_features_selected = 0

        while num_features_selected < len(X_train.columns):

            for i in range(X_train.shape[1]):
                if not features_mask_gbc[i]:
                    features_mask_gbc[i] = True

                    X_train_subset = X_train.iloc[:, features_mask_gbc]
                    X_test_subset = X_test.iloc[:, features_mask_gbc]

                    self.gbc.fit(X_train_subset, Y_train)
                    preds_gbc = self.gbc.predict(X_test_subset)
                    f1_gbc_subset = round(f1_score(Y_test, preds_gbc, average='weighted'), 3)
                    f1_gbc_list.append(f1_gbc_subset)
                    features_mask_gbc[i] = False
                else:
                    f1_gbc_list.append(-1) # That means it was previously present
                
                if not features_mask_rfc[i]:
                    features_mask_rfc[i] = True

                    X_train_subset = X_train.iloc[:, features_mask_rfc]
                    X_test_subset = X_test.iloc[:, features_mask_rfc]

                    self.rfc.fit(X_train_subset, Y_train)
                    preds_rfc = self.rfc.predict(X_test_subset)
                    f1_rfc_subset = round(f1_score(Y_test, preds_rfc, average='weighted'), 3)
                    f1_rfc_list.append(f1_rfc_subset)
                    features_mask_rfc[i] = False
                else:
                    f1_rfc_list.append(-1) # That means it was previously present

            if max(f1_gbc_list) >= f1_score_gbc_best or max(f1_rfc_list) >= f1_score_rfc_best:

                if max(f1_gbc_list) >= f1_score_gbc_best:
                    print(f"Best score gbc:, {max(f1_gbc_list)}, feature name:{X_train.columns[np.argmax(f1_gbc_list)]}")
                    best_features_gbc.add(X_train.columns[np.argmax(f1_gbc_list)])
                    gbc_score_improvement.append((max(f1_gbc_list), best_features_gbc))
                    features_mask_gbc[np.argmax(f1_gbc_list)] = True
                    f1_score_gbc_best = max(f1_gbc_list)

                if max(f1_rfc_list) >= f1_score_rfc_best:
                    print(f"Best score rfc: {max(f1_rfc_list)}, feature name:{X_train.columns[np.argmax(f1_rfc_list)]}")
                    best_features_rfc.add(X_train.columns[np.argmax(f1_rfc_list)])
                    rfc_score_improvement.append((max(f1_rfc_list), best_features_rfc))
                    features_mask_rfc[np.argmax(f1_rfc_list)] = True
                    f1_score_rfc_best = max(f1_rfc_list)
                
                f1_gbc_list.clear()
                f1_rfc_list.clear()
                num_features_selected += 1
            else:
                print("No improvements between previous score!")
                break

        metrics["GradientBoostClassifier"] = {"features" : best_features_gbc, "F1": f1_score_gbc_best}
        metrics["RandomForestClassifier"] = {"features": best_features_rfc, "F1": f1_score_rfc_best}

        return metrics, gbc_score_improvement, rfc_score_improvement
    
    def backward_elimination_method(self, X_train, X_test, Y_train, Y_test):
        best_features_rfc, best_features_gbc = set(X_train.columns), set(X_train.columns)
        metrics = {}
        f1_score_gbc_current_best = 0
        f1_score_rfc_current_best = 0
        f1_gbc_list = []
        f1_rfc_list = []
        gbc_score_improvement = []
        rfc_score_improvement = []
        features_mask_gbc = np.zeros(X_train.shape[1], dtype=bool)
        features_mask_rfc = np.zeros(X_train.shape[1], dtype=bool)
        num_features_selected = len(X_test.columns)

        self.gbc.fit(X_train, Y_train)
        self.rfc.fit(X_train, Y_train)

        preds_gbc = self.gbc.predict(X_test)
        preds_rfc = self.rfc.predict(X_test)

        f1_score_gbc_current_best = round(f1_score(Y_test, preds_gbc, average='weighted'), 3)
        f1_score_rfc_current_best = round(f1_score(Y_test, preds_rfc, average='weighted'), 3)

        while num_features_selected > 0:
            for i in range(X_train.shape[1]):
                if features_mask_gbc[i]:
                    features_mask_gbc[i] = False

                    X_train_subset = X_train.loc[:, features_mask_gbc]
                    X_test_subset = X_test.loc[:, features_mask_gbc]

                    self.gbc.fit(X_train_subset, Y_train)
                    preds_gbc = self.gbc.predict(X_test_subset)
                    f1_gbc_subset = round(f1_score(Y_test, preds_gbc, average='weighted'), 3)
                    f1_gbc_list.append(f1_gbc_subset)
                    print("f1 gbc subset:", f1_gbc_subset)

                    features_mask_gbc[i] = True
                else:
                    f1_gbc_list.append(-1) # That means it was removed already

                if features_mask_rfc[i]:
                    features_mask_rfc[i] = False

                    X_train_subset = X_train.loc[:, features_mask_rfc]
                    X_test_subset = X_test.loc[:, features_mask_rfc]

                    self.rfc.fit(X_train_subset, Y_train)
                    preds_rfc = self.rfc.predict(X_test_subset)
                    f1_rfc_subset = round(f1_score(Y_test, preds_rfc, average='weighted'), 3)
                    f1_rfc_list.append(f1_rfc_subset)
                    print("f1 rfc subset:", f1_rfc_subset)

                    features_mask_rfc[i] = True
                else:

                    f1_rfc_list.append(-1) # That means it was removed already

            if max(f1_gbc_list) >= f1_score_gbc_current_best or max(f1_rfc_list) >= f1_score_rfc_current_best:
                if max(f1_gbc_list) >= f1_score_gbc_current_best:
                    print("Best score gbc:", max(f1_gbc_list), X_train.columns[np.argmax(f1_gbc_list)])
                    best_features_gbc.remove(X_train.columns[np.argmax(f1_gbc_list)])
                    gbc_score_improvement.append((max(f1_gbc_list), best_features_gbc))
                    print("Best features gbc:", best_features_gbc)
                    features_mask_gbc[np.argmax(f1_gbc_list)] = False
                    f1_score_gbc_current_best = max(f1_gbc_list)
                    
                if max(f1_rfc_list) >= f1_score_rfc_current_best:
                    print("Best score rfc:", max(f1_rfc_list), X_train.columns[np.argmax(f1_rfc_list)])
                    best_features_rfc.remove(X_train.columns[np.argmax(f1_rfc_list)])
                    rfc_score_improvement.append((max(f1_rfc_list), best_features_rfc))
                    print("Best features rfc:", best_features_rfc)
                    features_mask_rfc[np.argmax(f1_rfc_list)] = False
                    f1_score_rfc_current_best = max(f1_rfc_list)
  
                f1_gbc_list.clear()
                f1_rfc_list.clear()
                num_features_selected -= 1

            else:
                print("No improvements between previous score!")
                break
        
        metrics["GradientBoostClassifier"] = {"features" : best_features_gbc, "F1": f1_score_gbc_current_best}
        metrics["RandomForestClassifier"] = {"features": best_features_rfc, "F1": f1_score_rfc_current_best}
        return metrics, gbc_score_improvement, rfc_score_improvement

    def rfe_method(self, X_train, X_test, Y_train, Y_test, limit_of_features) -> tuple:
        mutual_info = mutual_info_classif(X_train, y_train)
        mutual_info = pd.Series(mutual_info)
        mutual_info.index = X_train.columns
        mutual_info.sort_values(ascending=False).plot.bar()
        mutual_info = mutual_info.sort_values()
        current_features = len(mutual_info)
        progress_gbc = []
        progress_rfc = []
        
        metrics = {}

        while current_features > limit_of_features:
            mutual_info = mutual_info_classif(X=X_train, y=y_train)
            mutual_info = pd.Series(mutual_info, index=X_train.columns)
            print(f"Current features:{X_train.columns}")
            print(mutual_info.sort_values())
            print("Feature with lowest mutual information score:", mutual_info.idxmin())
            X_train = X_train.drop(columns=mutual_info.idxmin())
            X_test = X_test.drop(columns=mutual_info.idxmin())
            current_features = len(mutual_info)

            self.gbc.fit(X=X_train, y=y_train)
            self.rfc.fit(X=X_train, y=y_train)

            preds_gbc = self.gbc.predict(X_test)
            preds_rfc = self.rfc.predict(X_test)

            accuracy_gbc = round(accuracy_score(y_test, preds_gbc), 3)
            precision_gbc = round(precision_score(y_test, preds_gbc), 3)
            recall_gbc = round(recall_score(y_test, preds_gbc), 3)
            f1_gbc = round(f1_score(y_test, preds_gbc, average='weighted'), 3)

            print(f"Metrics for gradient boost classifier: accuracy:{accuracy_gbc}, precision:{precision_gbc}, recall:{recall_gbc}, f1:{f1_gbc}")

            accuracy_rfc = round(accuracy_score(y_test, preds_rfc), 3)
            precision_rfc = round(precision_score(y_test, preds_rfc), 3)
            recall_rfc = round(recall_score(y_test, preds_rfc), 3)
            f1_rfc = round(f1_score(y_test, preds_rfc, average='weighted'), 3)

            print(f"Metrics for random forest classifier: accuracy:{accuracy_rfc}, precision:{precision_rfc}, recall:{recall_rfc}, f1:{f1_rfc}")

            progress_gbc.append((X_train.columns, accuracy_gbc, precision_gbc, recall_gbc, f1_gbc))
            progress_rfc.append((X_train.columns, accuracy_rfc, precision_rfc, recall_rfc, f1_rfc))

        metrics["GradientBoostClassifier"] = {"features": X_train.columns, "Accuracy:" : accuracy_gbc, "Precision:" : precision_gbc, "Recall:" : recall_gbc, "F1 Score:" : f1_gbc}
        metrics["RandomForestClassifier"] = {"features": X_train.columns, "Accuracy:" : accuracy_rfc, "Precision:" : precision_rfc, "Recall:" : recall_rfc, "F1 Score:" : f1_rfc}

        return metrics, progress_gbc, progress_rfc

    def initialize_population(self, population_size, num_features) -> list:
        population = self.rng.integers(2, size=(population_size, num_features))
        print("Population generated!\n")
        return population

    def calculate_fitness(self, features: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, Y_test: pd.DataFrame) -> tuple:
        feature_mask = features.astype(bool)
        X_train_subset = X_train.loc[:, feature_mask]
        X_test_subset = X_test.loc[:, feature_mask]

        self.gbc.fit(X_train_subset, y_train)
        self.rfc.fit(X_train_subset, y_train)

        preds_gbc = self.gbc.predict(X_test_subset)
        preds_rfc = self.rfc.predict(X_test_subset)

        f1_score_gbc_subset = round(f1_score(Y_test, preds_gbc, average='weighted'), 3)
        f1_score_rfc_subset = round(f1_score(Y_test, preds_rfc, average='weighted'), 3)
        
        return f1_score_gbc_subset, f1_score_rfc_subset, features

    def calculate_individual_probabilities(self, individuals, fitness_scores_list, fitness_scores_sum) -> list:
        for i in range(len(fitness_scores_list)):
            probability = fitness_scores_list[i][0]/fitness_scores_sum
            individuals[i] = probability
        return individuals

    def selection_roulette_ga(self, population, population_size, output_prob: dict) -> list:
        half_population_size = int(population_size/2)
        indices = list(output_prob.keys())
        selected_indices = self.rng.choice(indices, half_population_size, p=list(output_prob.values()))
        selected_population = population[selected_indices]
        return selected_population

    def selection_tournament_ga(self, population, tournament_size, fitness_score_dict) -> None:
        num_of_tournaments = len(population)//tournament_size
        print("Number of tournament:", num_of_tournaments)
        selected_population = []
        list_of_winners = []
        indices = list(fitness_score_dict.keys())

        for _ in range(num_of_tournaments):
            selected_indices = self.rng.choice(indices, tournament_size, replace=False)
            tournament_dict = {k: fitness_score_dict[k][0] for k in selected_indices}
            winner = max(tournament_dict, key=lambda k: tournament_dict[k])
            list_of_winners.append(winner - 1) # To correct index

        selected_population = population[list_of_winners]
        return selected_population
    
    def uniform_crossover(self, selected_population, population_size) -> None:
        next_gen = []
        i = population_size

        for _ in tqdm(range(i), desc='Crossover progress', colour = "green", leave=True):
            random_row_indices = self.rng.choice(selected_population.shape[0], size=2, replace=False)
            random_parents = selected_population[random_row_indices, :]
            offspring = np.zeros((1, random_parents.shape[1]))

            for j in range(random_parents.shape[1]):
                offspring[:, j] = self.rng.choice(random_parents[:, j], size=1)

            for k in range(offspring.shape[0]):
                next_gen.append(offspring[k])
        
        next_gen = np.array(next_gen, dtype=int)

        return next_gen
    
    def mutation(self, next_gen, selected_population):
        for i in range(next_gen.shape[0]):
           random_gene_from_individual = self.rng.choice(selected_population.shape[1], size=1)
           if next_gen[i][random_gene_from_individual] == 1:
              next_gen[i][random_gene_from_individual] = 0
           else:
              next_gen[i][random_gene_from_individual] = 1
        return next_gen

    # Powinienem dac mozliwosc wyboru selekcji (jakas flaga)
    def run_genethic_algo(self, population_size, num_of_generations, features, filepath, target_col) -> tuple:
        best_fitness_score = 0
        generations_fitness = {}
         
        data = self.prepare_data(filepath=filepath)
    
        X, Y, X_train, X_test, Y_train, Y_test = self.split_data(data=data, target_col=target_col)
    
        population = self.initialize_population(population_size, features)
    
        for i in range (num_of_generations):
            print("Fitness calculation !")
            fitness_scores_list = [self.calculate_fitness(features, X_train, Y_train, X_test, Y_test) for features in tqdm(population, desc='Fitness Score Calculation Progress', colour = "green", leave=True)]

            if np.max([score[0] for score in fitness_scores_list]) >= best_fitness_score:
                best_fitness_score = np.max([score[0] for score in fitness_scores_list]) # gbc F1 [0], rfc F1 [1]
                best_individual = fitness_scores_list[np.argmax([score[0] for score in fitness_scores_list])][2] # index of best fitness score and second element of tuple cause it is binary mask
                
        
            fitness_scores_dict = {i + 1: score for i, score in enumerate(fitness_scores_list)}
            print("DICT:", fitness_scores_dict)
            print("\nNajlepszy wynik dla generacji nr :", i+1 , "to :", np.max([score[0] for score in fitness_scores_list]))
            generations_fitness[i+1] = {"Score": np.max([score[0] for score in fitness_scores_list])}
            fitness_scores_sum = sum(score[0] for score in fitness_scores_list)
            individ = {}
            output_prob = self.calculate_individual_probabilities(individ, fitness_scores_list, fitness_scores_sum)
            #selected_population = self.selection_roulette_ga(population , population_size, output_prob)
            selected_population = self.selection_tournament_ga(population, 3, fitness_scores_dict)
            next_gen = self.uniform_crossover(selected_population, population_size)
            next_gen_mutated = self.mutation(next_gen, selected_population)
            population = next_gen_mutated
            print("Najlepszy fitness score", best_fitness_score)

        mask = best_individual.astype(bool)
        best_features = X_test.loc[:, mask]
        selected_features = best_features.columns.to_list()
        return best_fitness_score, selected_features, generations_fitness

    def generate_pheromone_array(self, num_features, value) -> None:
        pheromone_array = np.full(num_features, value)
        return pheromone_array
    
    def generate_ant_population(self, ants_colony_size, num_features) -> None:
        ants = np.zeros(shape=(ants_colony_size, num_features), dtype=int)
        for ant in ants:
            random_node_index = self.rng.integers(0, num_features-1)
            ant[random_node_index] = 1
        print("Ants generated!\n")
        return ants

    def roulette_wheel_selection_aco(self, probabilities, num_candidates) -> None:
        print("\nProblities:\n", probabilities)
        cumulative_probabilities = list(accumulate(probabilities, operator.add))
        print("\nCumulative probabilities: \n", cumulative_probabilities)
        selected_index = np.searchsorted(cumulative_probabilities, np.random.rand()) % num_candidates
        return selected_index

    def ants_make_move(self, ants_population, pheromone_array, weights, alpha, beta) -> None:
        for ant_index in range(len(ants_population)):
            current_ant = ants_population[ant_index]

            print("\nAnt {}:".format(ant_index + 1))
            print(current_ant)

            # Calculate the sum of probabilities for normalization
            total_probability = np.sum((pheromone_array[current_ant == 0] ** alpha) * (weights[current_ant == 0] ** beta))
            probabilities_for_each_candidate = np.zeros(len(current_ant))

            for j in range(len(current_ant)):
                if current_ant[j] == 0:
                    pheromone_level = pheromone_array[j]
                    attractiveness = weights[j]
                    probability = ((pheromone_level ** alpha) * (attractiveness ** beta)) / total_probability
                    probabilities_for_each_candidate[j] = probability

            selected_feature_index = self.roulette_wheel_selection_aco(probabilities_for_each_candidate, len(probabilities_for_each_candidate))
            print("Selected Feature Index:", selected_feature_index)
            print()
            current_ant[selected_feature_index] = 1
            print("Ant selected:\n", current_ant)
            print()

        return ants_population

    def update_pheromones(self, pheromone_array, ants_population: list, quality_measure, evaporation_rate=0.1) -> None:
        pheromone_array *= (1 - evaporation_rate)
        print(ants_population)
        for ant_index, ant in enumerate(ants_population):
            for j in range(len(ant)):
                print(ant[j])
                if ant[j] == 1:
                    pheromone_array[j] += quality_measure[ant_index][0]
        return pheromone_array

    def aco_run(self, X_train, X_test, Y_train, Y_test, base_pheromone_value, ants_colony_size, ant_generations, ants_moves, alpha, beta) -> tuple:
        mutual_info = mutual_info_classif(X_train, Y_train)
        best_fitness_score = 0
        weights = np.array(mutual_info)
        print(len(mutual_info))
        pheromone_array = self.generate_pheromone_array(len(mutual_info), base_pheromone_value)
        print("\nWagi:\n",weights)
        print("\nInitial pheromone_array: \n", pheromone_array)


        for _ in range(ant_generations):
            ants = self.generate_ant_population(ants_colony_size, len(mutual_info))
            for _ in range(ants_moves): 
                ants_population_after_move = self.ants_make_move(ants, pheromone_array=pheromone_array, weights=weights, alpha=alpha, beta=beta)
                print(ants_population_after_move)
                fitness_scores_ants = [self.calculate_fitness(features, X_train, Y_train, X_test, Y_test) for features in tqdm(ants_population_after_move, desc='Fitness Score Calculation Progress', colour = "green", leave=True)]

                if np.max([score[0] for score in fitness_scores_ants]) >= best_fitness_score:
                    best_fitness_score = np.max([score[0] for score in fitness_scores_ants]) # gbc F1 [0], rfc F1 [1]
                    best_individual = fitness_scores_ants[np.argmax([score[0] for score in fitness_scores_ants])][2] # index of best fitness score and second element of tuple cause it is binary mask

                updated_pheromones = self.update_pheromones(pheromone_array, ants_population_after_move, fitness_scores_ants)
                print("\nUpdated Pheromone Levels:\n", updated_pheromones)
                print("\nNew ants population\n:", ants_population_after_move)

        return best_individual, best_fitness_score

instance = WrapperMethods()
data = instance.prepare_data("/Users/patrykjaworski/Documents/Projekty/Feature-Selection/TESTY/Old/data.csv")
print(data)
X, Y, X_train, X_test, y_train, y_test = instance.split_data(data, "diagnosis")
# best_columns, improvement_gbc, improvement_rfc = instance.forward_selection_method(X_train, X_test, y_train, y_test)
# print(f"Best columns: {best_columns}, improvement_gbc: {improvement_gbc}, improvement_rfc: {improvement_rfc}")
# best_columns, improvement_gbc, improvement_rfc = instance.backward_elimination_method(X_train, X_test, y_train, y_test)
# print(f"Best columns: {best_columns} \n improvement_gbc: {improvement_gbc} \n improvement_rfc: {improvement_rfc}")
#best_score, best_osobnik, generations_dict = instance.run_genethic_algo(30, 5, 30,"/Users/patrykjaworski/Documents/Projekty/Feature-Selection/TESTY/Old/data.csv",'diagnosis')
# best_individ, score = instance.aco_run(X_train, X_test, y_train, y_test, base_pheromone_value=0.25, ants_colony_size=3, ant_generations=3, ants_moves=5, alpha=1, beta=2)
# print(best_individ, score)
metrics, improvement_gbc, improvement_rfc = instance.rfe_method(X_train, X_test, y_train, y_test, 18)

