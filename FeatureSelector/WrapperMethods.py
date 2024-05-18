import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
import numpy as np
from tqdm import tqdm
from plotly.express import line
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from numba import jit
from itertools import accumulate
import operator
import matplotlib.pyplot as plt
import networkx as nx
import json
from timeit import default_timer as timer

class WrapperMethods:
    def __init__(self) -> None:
        self.gbc = GradientBoostingClassifier()
        self.rfc = RandomForestClassifier()
        self.rng = np.random.default_rng()
        
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
        return X, Y

    def forward_selection_method(self, X, Y) -> tuple:
        best_features_rfc, best_features_gbc = set(), set()
        metrics = {}
        f1_score_gbc_best = 0
        f1_score_rfc_best = 0
        f1_gbc_list = []
        f1_rfc_list = []
        gbc_score_improvement = []
        rfc_score_improvement = []
        features_mask_gbc = np.zeros(X.shape[1], dtype=bool)
        features_mask_rfc = np.zeros(X.shape[1], dtype=bool)
        num_features_selected = 0

        start = timer()
        while num_features_selected < X.shape[1]:

            for i in range(X.shape[1]):
                if not features_mask_gbc[i]:
                    features_mask_gbc[i] = True

                    X_subset = X.iloc[:, features_mask_gbc]
                    f1_gbc_subset = np.mean(cross_val_score(self.gbc, X_subset, Y, cv=5, scoring='f1_weighted'))
                    print("F1_GBC_SUBSET:", f1_gbc_subset)
                    
                    f1_gbc_list.append(f1_gbc_subset)
                    features_mask_gbc[i] = False
                else:
                    f1_gbc_list.append(-1) # That means it was previously present
                
                if not features_mask_rfc[i]:
                    features_mask_rfc[i] = True

                    X_subset = X.iloc[:, features_mask_rfc]

                    f1_rfc_subset = np.mean(cross_val_score(self.rfc, X_subset, Y, cv=5, scoring='f1_weighted'))
                    print("F1_RFC_SUBSET:", f1_rfc_subset)
                    f1_rfc_list.append(f1_rfc_subset)

                    features_mask_rfc[i] = False
                else:
                    f1_rfc_list.append(-1) # That means it was previously present

            if max(f1_gbc_list) >= f1_score_gbc_best or max(f1_rfc_list) >= f1_score_rfc_best:

                if max(f1_gbc_list) >= f1_score_gbc_best:
                    print(f"Best score gbc:, {max(f1_gbc_list)}, feature name:{X.columns[np.argmax(f1_gbc_list)]}")
                    best_features_gbc.add(X.columns[np.argmax(f1_gbc_list)])
                    gbc_score_improvement.append((max(f1_gbc_list), set(best_features_gbc)))
                    print(f"Current best feautures gbc:{best_features_gbc}\n\n")
                    features_mask_gbc[np.argmax(f1_gbc_list)] = True
                    f1_score_gbc_best = max(f1_gbc_list)
                else:
                    print("No improvement in GBC!")

                if max(f1_rfc_list) >= f1_score_rfc_best:
                    print(f"Best score rfc: {max(f1_rfc_list)}, feature name:{X.columns[np.argmax(f1_rfc_list)]}")
                    best_features_rfc.add(X.columns[np.argmax(f1_rfc_list)])
                    rfc_score_improvement.append((max(f1_rfc_list), set(best_features_rfc)))
                    print(f"Current best feautures rfc:{best_features_rfc}\n\n")
                    features_mask_rfc[np.argmax(f1_rfc_list)] = True
                    f1_score_rfc_best = max(f1_rfc_list)
                else:
                    print("No improvement in RFC!")
                
                f1_gbc_list.clear()
                f1_rfc_list.clear()
                num_features_selected += 1
            else:
                print("No improvements between previous score!")
                end = timer()
                print(f"Feature selection took: {end-start} s.")
                break

        progress_gbc = {
            "f1_score_gbc": [score for score, _ in gbc_score_improvement],
            "features_num": [len(features) for _, features in gbc_score_improvement],
            "features_gbc": [str(features) for _, features in gbc_score_improvement],
        }

        progress_rfc = {
            "f1_score_rfc": [score for score, _ in rfc_score_improvement],
            "features_num": [len(features) for _, features in rfc_score_improvement],
            "features_rfc": [str(features) for _, features in rfc_score_improvement],

        }

        df_gbc = pd.DataFrame(data=progress_gbc)
        df_rfc = pd.DataFrame(data=progress_rfc)
        df_gbc.index += 1
        df_rfc.index += 1

        fig = line(df_gbc["f1_score_gbc"], title='Fitness forward selection over iterations')
        fig.write_html("fitness_forward_selection.html")

        with pd.ExcelWriter('progress_forward_selection.xlsx') as writer:  
            df_gbc.to_excel(writer, sheet_name='GBC')
            df_rfc.to_excel(writer, sheet_name='RFC')

        metrics["GradientBoostClassifier"] = {"features" : best_features_gbc, "F1": f1_score_gbc_best}
        metrics["RandomForestClassifier"] = {"features": best_features_rfc, "F1": f1_score_rfc_best}

        return metrics, gbc_score_improvement, rfc_score_improvement
    
    def backward_elimination_method(self, X, Y):
        best_features_rfc, best_features_gbc = set(X.columns), set(X.columns)
        metrics = {}
        f1_score_gbc_current_best = 0
        f1_score_rfc_current_best = 0
        f1_gbc_list = []
        f1_rfc_list = []
        gbc_score_improvement = []
        rfc_score_improvement = []
        features_mask_gbc = np.ones(X.shape[1], dtype=bool)
        features_mask_rfc = np.ones(X.shape[1], dtype=bool)
        num_features_selected = len(X.columns)

        f1_score_gbc_current_best = np.mean(cross_val_score(self.gbc, X, Y, cv=5, scoring='f1_weighted'))
        f1_score_rfc_current_best = np.mean(cross_val_score(self.rfc, X, Y, cv=5, scoring='f1_weighted'))

        start = timer()
        while num_features_selected > 0:
            for i in range(X.shape[1]):
                if features_mask_gbc[i]:
                    features_mask_gbc[i] = False

                    X_subset = X.iloc[:, features_mask_gbc]

                    f1_gbc_subset = np.mean(cross_val_score(self.gbc, X_subset, Y, cv=5, scoring='f1_weighted'))
                    print(f"GBC_SUBSET:{f1_gbc_subset}")
                    f1_gbc_list.append(f1_gbc_subset)

                    features_mask_gbc[i] = True
                else:
                    f1_gbc_list.append(-1) # That means it was removed already

                if features_mask_rfc[i]:
                    features_mask_rfc[i] = False

                    X_subset = X.iloc[:, features_mask_rfc]
 
                    f1_rfc_subset = np.mean(cross_val_score(self.gbc, X_subset, Y, cv=5, scoring='f1_weighted'))
                    print(f"RFC_SUBSET:{f1_rfc_subset}")
                    f1_rfc_list.append(f1_rfc_subset)

                    features_mask_rfc[i] = True
                else:
                    f1_rfc_list.append(-1) # That means it was removed already

            if max(f1_gbc_list) >= f1_score_gbc_current_best or max(f1_rfc_list) >= f1_score_rfc_current_best:
                if max(f1_gbc_list) >= f1_score_gbc_current_best:
                    print("Best score gbc:", max(f1_gbc_list), X.columns[np.argmax(f1_gbc_list)])
                    best_features_gbc.remove(X.columns[np.argmax(f1_gbc_list)])
                    gbc_score_improvement.append((max(f1_gbc_list), set(best_features_gbc)))
                    print(f"Current best features gbc: {best_features_gbc}\n", )
                    features_mask_gbc[np.argmax(f1_gbc_list)] = False
                    f1_score_gbc_current_best = max(f1_gbc_list)
                    
                if max(f1_rfc_list) >= f1_score_rfc_current_best:
                    print("Best score rfc:", max(f1_rfc_list), X.columns[np.argmax(f1_rfc_list)])
                    best_features_rfc.remove(X.columns[np.argmax(f1_rfc_list)])
                    rfc_score_improvement.append((max(f1_rfc_list), set(best_features_rfc)))
                    print(f"Current best features rfc: {best_features_rfc}\n")
                    features_mask_rfc[np.argmax(f1_rfc_list)] = False
                    f1_score_rfc_current_best = max(f1_rfc_list)
  
                f1_gbc_list.clear()
                f1_rfc_list.clear()
                num_features_selected -= 1

            else:
                print("No improvements between previous score!")
                end=timer()
                print(f"Feature selection took: {end-start} s.")
                break

        progress_gbc = {
            "f1_score_gbc": [score for score, _ in gbc_score_improvement],
            "features_num": [len(features) for _, features in gbc_score_improvement],
            "features_gbc": [str(features) for _, features in gbc_score_improvement],
        }

        progress_rfc = {
            "f1_score_rfc": [score for score, _ in rfc_score_improvement],
            "features_num": [len(features) for _, features in rfc_score_improvement],
            "features_rfc": [str(features) for _, features in rfc_score_improvement],

        }

        df_gbc = pd.DataFrame(data=progress_gbc)
        df_rfc = pd.DataFrame(data=progress_rfc)
        df_gbc.index += 1
        df_rfc.index += 1

        fig = line(df_gbc["f1_score_gbc"], title='Fitness backward elimination over iterations')
        fig.write_html("fitness_backward_elimination.html")

        with pd.ExcelWriter('progress_backward_elimination.xlsx') as writer:  
            df_gbc.to_excel(writer, sheet_name='GBC')
            df_rfc.to_excel(writer, sheet_name='RFC')

        metrics["GradientBoostClassifier"] = {"features" : best_features_gbc, "F1": f1_score_gbc_current_best}
        metrics["RandomForestClassifier"] = {"features": best_features_rfc, "F1": f1_score_rfc_current_best}
        return metrics, gbc_score_improvement, rfc_score_improvement

    def rfe_method(self, X, Y, limit_of_features) -> tuple:
        mutual_info = mutual_info_classif(X, Y)
        mutual_info = pd.Series(mutual_info)
        mutual_info.index = X.columns
        mutual_info.sort_values(ascending=False).plot.bar()
        mutual_info = mutual_info.sort_values()
        current_features = len(mutual_info)
        progress_gbc = []
        progress_rfc = []
        
        metrics = {}

        start = timer()

        while current_features > limit_of_features:
            mutual_info = mutual_info_classif(X=X, y=Y)
            mutual_info = pd.Series(mutual_info, index=X.columns)
            print(f"Current features:{X.columns}")
            print(mutual_info.sort_values())
            print("Feature with lowest mutual information score:", mutual_info.idxmin())
            X = X.drop(columns=mutual_info.idxmin())
            current_features = len(mutual_info)

            accuracy_gbc = np.mean(cross_val_score(self.gbc, X, Y, cv=5, scoring='accuracy'))
            precision_gbc = np.mean(cross_val_score(self.gbc, X, Y, cv=5, scoring='precision'))
            recall_gbc = np.mean(cross_val_score(self.gbc, X, Y, cv=5, scoring='recall'))
            f1_gbc = np.mean(cross_val_score(self.gbc, X, Y, cv=5, scoring='f1_weighted'))

            print(f"Metrics for gradient boost classifier: accuracy:{accuracy_gbc}, precision:{precision_gbc}, recall:{recall_gbc}, f1:{f1_gbc}")

            accuracy_rfc = np.mean(cross_val_score(self.rfc, X, Y, cv=5, scoring='accuracy'))
            precision_rfc = np.mean(cross_val_score(self.rfc, X, Y, cv=5, scoring='precision'))
            recall_rfc = np.mean(cross_val_score(self.rfc, X, Y, cv=5, scoring='recall'))
            f1_rfc = np.mean(cross_val_score(self.rfc, X, Y, cv=5, scoring='f1_weighted'))

            print(f"Metrics for random forest classifier: accuracy:{accuracy_rfc}, precision:{precision_rfc}, recall:{recall_rfc}, f1:{f1_rfc}")

            progress_gbc.append((X.columns, accuracy_gbc, precision_gbc, recall_gbc, f1_gbc))
            progress_rfc.append((X.columns, accuracy_rfc, precision_rfc, recall_rfc, f1_rfc))
        
        end = timer()
        print(f"Feature selection took: {end-start} s.")

        gbc_data = {
            "accuracy": [accuracy for _, accuracy, _, _, _  in progress_gbc],
            "precision": [precision for _, _, precision, _, _ in progress_gbc],
            "recall": [recall for _, _, _, recall, _ in progress_gbc],
            "f1": [f1_best for _, _, _, _, f1_best in progress_gbc],
            "features_num": [len(features) for features, _, _, _, _ in progress_gbc],
            "features": [str(features) for features, _, _, _, _ in progress_gbc],
        }

        rfc_data = {
            "accuracy": [accuracy for _, accuracy, _, _, _  in progress_rfc],
            "precision": [precision for _, _, precision, _, _ in progress_rfc],
            "recall": [recall for _, _, _, recall, _ in progress_rfc],
            "f1": [f1_best for _, _, _, _, f1_best in progress_rfc],
            "features_num": [len(features) for features, _, _, _, _ in progress_rfc],
            "features": [features for features, _, _, _, _ in progress_rfc],
        }

        df_gbc = pd.DataFrame(data=gbc_data)
        df_rfc = pd.DataFrame(data=rfc_data)
        df_gbc.index += 1
        df_rfc.index += 1

        fig = line(df_gbc["f1"], title='Fitness RFE over iterations')
        fig.write_html("fitness_rfe.html")

        with pd.ExcelWriter('progress_rfe.xlsx') as writer:
            df_gbc.to_excel(writer, sheet_name='GBC')
            df_rfc.to_excel(writer, sheet_name='RFC')

        metrics["GradientBoostClassifier"] = {"features": X.columns, "Accuracy:" : accuracy_gbc, "Precision:" : precision_gbc, "Recall:" : recall_gbc, "F1 Score:" : f1_gbc}
        metrics["RandomForestClassifier"] = {"features": X.columns, "Accuracy:" : accuracy_rfc, "Precision:" : precision_rfc, "Recall:" : recall_rfc, "F1 Score:" : f1_rfc}

        return metrics, progress_gbc, progress_rfc

    def initialize_population(self, population_size, num_features) -> list:
        population = self.rng.integers(2, size=(population_size, num_features))
        print("Population generated!\n")
        return population

    def calculate_fitness(self, features: pd.DataFrame, X: pd.DataFrame, Y: pd.DataFrame) -> tuple:
        feature_mask = features.astype(bool)
        X_subset = X.loc[:, feature_mask]

        f1_score_gbc_subset = np.mean(cross_val_score(self.gbc, X_subset, Y, cv=5, scoring='f1_weighted'))
        f1_score_rfc_subset = np.mean(cross_val_score(self.rfc, X_subset, Y, cv=5, scoring='f1_weighted'))
        
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
        generations_best_fitness = {}
        generations_individuals = []
        list_individuals = []
         
        data = self.prepare_data(filepath=filepath)
    
        X, Y = self.split_data(data=data, target_col=target_col)
    
        population = self.initialize_population(population_size, features)
    
        for i in range (num_of_generations):
            print("Fitness calculation !")
            fitness_scores_list = [self.calculate_fitness(features, X, Y) for features in tqdm(population, desc='Fitness Score Calculation Progress', colour = "green", leave=True)]

            if np.max([score[0] for score in fitness_scores_list]) >= best_fitness_score:
                best_fitness_score = np.max([score[0] for score in fitness_scores_list]) # gbc F1 [0], rfc F1 [1]
                best_individual = fitness_scores_list[np.argmax([score[0] for score in fitness_scores_list])][2] # index of best fitness score and second element of tuple cause it is binary mask
            
            fitness_scores_dict = {i + 1: score for i, score in enumerate(fitness_scores_list)}
            generations_individuals.append(fitness_scores_dict)
            print("DICT:", fitness_scores_dict)
            print("\nNajlepszy wynik dla generacji nr :", i+1 , "to :", np.max([score[0] for score in fitness_scores_list]))


            mask = fitness_scores_list[np.argmax([score[0] for score in fitness_scores_list])][2].astype(bool)
            best_features = X.loc[:, mask]
            
            generations_best_fitness[i+1] = {"Score GBC": np.max([score[0] for score in fitness_scores_list]), 
                                        "Score RFC": np.max([score[1] for score in fitness_scores_list]),
                                        "Best features:": best_features.columns.to_list()
                                       }
            
            print("GENERATIONS FITNESS: ", generations_best_fitness)
            
            fitness_scores_sum = sum(score[0] for score in fitness_scores_list)
            individ = {}
            output_prob = self.calculate_individual_probabilities(individ, fitness_scores_list, fitness_scores_sum)
            selected_population = self.selection_roulette_ga(population , population_size, output_prob)
            #selected_population = self.selection_tournament_ga(population, 3, fitness_scores_dict)
            next_gen = self.uniform_crossover(selected_population, population_size)
            next_gen_mutated = self.mutation(next_gen, selected_population)
            population = next_gen_mutated
            print("Najlepszy fitness score", best_fitness_score)

        for generation_data in generations_individuals:
            fitness_values = [fitness_data[0] for fitness_data in generation_data.values()]
            list_individuals.append(fitness_values)

        df_generations = pd.DataFrame(list_individuals).transpose()
        df_generations.columns = [f"generation {i}" for i in range(1, len(list_individuals) + 1)]
        df_generations.index = [f"individal {i}" for i in range(1, len(list_individuals[0]) + 1)]

        df_generations.to_excel("individuals_over_generations.xlsx", index=True)

        df_best_generations = pd.DataFrame(data=generations_best_fitness).T
        df_best_generations.to_excel("best_generations.xlsx", index=True)
        print(df_best_generations)

        fig = line(df_generations, title='Fitness over Generations')
        fig.write_html("fitness_over_generations.html")

        fig2 = line(df_best_generations, x=df_best_generations.index, y=['Score GBC', 'Score RFC'], title='Best fitness over generations')
        fig2.write_html("best_fitness_over_generations.html")
        return best_fitness_score, best_individual

    def generate_pheromone_array(self, num_features, value) -> None:
        pheromone_array = np.full(num_features, value)
        return pheromone_array
    
    def generate_ant_population(self, ants_colony_size, num_features, history_of_moves: list, generation) -> tuple:
        ants = np.zeros(shape=(ants_colony_size, num_features), dtype=int)
        generation_data = []
        for ant in ants:
            random_node_index = self.rng.integers(0, num_features-1)
            ant[random_node_index] = 1
            generation_data.append([random_node_index+1])

        if not history_of_moves:
            history_of_moves.append({})
        elif generation + 1 not in history_of_moves[0]:
            history_of_moves[0][generation + 1] = []
          
        history_of_moves[0][generation + 1] = generation_data

        return ants, history_of_moves

    def roulette_wheel_selection_aco(self, probabilities, num_candidates) -> int:
        cumulative_probabilities = list(accumulate(probabilities, operator.add))
        selected_index = np.searchsorted(cumulative_probabilities, self.rng.random()) % num_candidates
        return selected_index

    def ants_make_move(self, ants_population, pheromone_array, weights, generation, history_of_moves: list, alpha, beta) -> tuple:
        for ant_index in range(len(ants_population)):
            current_ant = ants_population[ant_index]

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
            current_ant[selected_feature_index] = 1
            # Append the selected feature index to the sublist corresponding to the current ant index under the key representing the generation
            history_of_moves[0][generation+1][ant_index].append(selected_feature_index+1)

        return ants_population, history_of_moves

    def update_pheromones(self, pheromone_array, ants_population: list, quality_measure, evaporation_rate=0.1) -> None:
        pheromone_array *= (1 - evaporation_rate)

        for ant_index, ant in enumerate(ants_population):
            for j in range(len(ant)):
                if ant[j] == 1:
                    pheromone_array[j] += quality_measure[ant_index][0]
        return pheromone_array

    def aco_run(self, X, Y, base_pheromone_value, ants_colony_size, ant_generations, ants_moves, alpha, beta) -> tuple:
        start = timer()
        mutual_info = mutual_info_classif(X, Y)
        best_fitness_score = 0
        history = []
        best_ant = []
        best_ants =[]

        weights = np.array(mutual_info)
        pheromone_array = self.generate_pheromone_array(len(mutual_info), base_pheromone_value)

        for generation in range(ant_generations):
            ants, history_of_moves = self.generate_ant_population(ants_colony_size, len(mutual_info), history, generation)
            for _ in range(ants_moves): 
                ants_population_after_move, history_of_moves = self.ants_make_move(ants, pheromone_array=pheromone_array, weights=weights, generation=generation, history_of_moves=history_of_moves, alpha=alpha, beta=beta)
                print("Ant population after move:", ants_population_after_move)
                fitness_scores_ants = [self.calculate_fitness(features, X, Y) for features in tqdm(ants_population_after_move, desc='Fitness Score Calculation Progress', colour = "green", leave=False)]

                if np.max([score[0] for score in fitness_scores_ants]) >= best_fitness_score:
                    best_fitness_index = np.argmax([score[0] for score in fitness_scores_ants])
                    best_fitness_score = np.max([score[0] for score in fitness_scores_ants]) # gbc F1 [0], rfc F1 [1]
                    best_ant = fitness_scores_ants[best_fitness_index][2].copy() # index of best fitness score and second element of tuple cause it is binary mask
                    best_ant_mask = best_ant.astype(bool)
                    best_ant_path = history_of_moves[0][generation + 1][best_fitness_index].copy()
                    print(f"Best fitness index: {best_fitness_index}\n Best score: {best_fitness_score}\n Best ant path: {best_ant_path}\n Best ant mask:{best_ant}")
                    best_ants.append((best_fitness_score, best_ant, best_ant_path, X.loc[:, best_ant_mask].columns, len(best_ant_path)))
                    
                self.update_pheromones(pheromone_array, ants_population_after_move, fitness_scores_ants)
                history = history_of_moves
                print(history)
            print("Best ant:", best_ant)

        end = timer()
        print(f"Feature selection with Ant Colony Optimization took: {end-start}s.")

        df_ants = pd.DataFrame(best_ants, columns=["best_fitness_score", "best_ant", "best_ant_path", "features", "features_num"])
        df_ants.index += 1
        df_ants.to_excel("best_ants.xlsx", index=True)
        fig = line(df_ants['best_fitness_score'], title='Fitness of best ants during feature selection')
        fig.write_html("fitness_over_ants.html")
        edges = [(best_ant_path[i], best_ant_path[i+1]) for i in range(len(best_ant_path) - 1)]

        G = nx.DiGraph(directed=True)
        nx.circular_layout(G)
        G.add_edges_from(edges)
        nx.draw_networkx(G, with_labels=True, node_color='skyblue', arrowsize=20)
        plt.savefig("best_ant_graph.png")

        return best_ant, best_ant_path, best_fitness_score, history

instance = WrapperMethods()
data = instance.prepare_data("/Users/patrykjaworski/Documents/Projekty/Feature-Selection/TESTY/Old/data.csv")
print(data)
X, Y = instance.split_data(data, "diagnosis")
#best_columns, improvement_gbc, improvement_rfc = instance.forward_selection_method(X, Y)
# print(f"Best columns: {best_columns}, improvement_gbc: {improvement_gbc}, improvement_rfc: {improvement_rfc}")
#best_columns, improvement_gbc, improvement_rfc = instance.backward_elimination_method(X, Y)
# print(f"Best columns: {best_columns} \n improvement_gbc: {improvement_gbc} \n improvement_rfc: {improvement_rfc}")
#best_score, best_osobnik = instance.run_genethic_algo(15, 50, 30,"/Users/patrykjaworski/Documents/Projekty/Feature-Selection/TESTY/Old/data.csv",'diagnosis')
# print(best_score, best_osobnik)
ant, ant_path, score, history = instance.aco_run(X, Y, base_pheromone_value=0.25, ants_colony_size=30, ant_generations=4, ants_moves=30, alpha=1, beta=2)
#print(ant, ant_path, score)
#metrics, improvement_gbc, improvement_rfc = instance.rfe_method(X, Y, 5)

