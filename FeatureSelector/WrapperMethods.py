import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, mean_squared_error, confusion_matrix
from numba import jit
from itertools import accumulate
import operator


class WrapperMethods:
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
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        return X, Y, X_train, X_test, y_train, y_test

    def forward_selection_method(self, X_train, X_test, Y_train, Y_test) -> None:     
        best_features = set()
        scores = []
        features_mask = np.zeros(X_train.shape[1], dtype=bool)
        best_score = 0
        f1_score_best = 0

        while np.sum(features_mask) < 30:
            for i in range(X_train.shape[1]):
                if not features_mask[i]:
                    features_mask[i] = True
                    X_train_subset = X_train.loc[:, features_mask]
                    X_test_subset = X_test.loc[:, features_mask]
                    self.gbc.fit(X_train_subset, Y_train)
                    preds = self.gbc.predict(X_test_subset)
                    f1_score_subset = round(f1_score(Y_test, preds, average='weighted'), 3)
                    scores.append(f1_score_subset)
                    features_mask[i] = False
                else:
                    scores.append(0)
            print(scores)
            best_index = np.argmax(scores)
            best_score = max(scores)
            if best_score >= f1_score_best:
                print("Best index", best_index)
                best_features.add(X_train.columns[best_index])
                features_mask[best_index] = True
                print("\nNew Features Mask: \n", features_mask)
                X_train_best = X_train.loc[:, features_mask]
                X_test_best = X_test.loc[:, features_mask]
                self.gbc.fit(X_train_best, Y_train)
                preds = self.gbc.predict(X_test_best)
                f1_score_best = round(f1_score(Y_test, preds, average='weighted'), 3)
                print(f1_score_best)
                print("\nbest_features", best_features, "\n")
                scores.clear()
            else:
                print("\nNo improvment between previous score !\n")
                print("\nbest_features", best_features, "\n")
                break

    def backward_elimination_method(self, X_train, X_test, Y_train, Y_test) -> None:
        best_features = []
        scores = []
        features_mask = np.ones(X_train.shape[1], dtype=bool)
        original_indices = list(range(X_train.shape[1]))
        X_train_subset = X_train.loc[:, features_mask]
        X_test_subset = X_test.loc[:, features_mask]
        self.gbc.fit(X_train_subset, Y_train)
        preds = self.gbc.predict(X_test_subset)
        f1_score_features = round(f1_score(Y_test, preds, average='weighted'), 3)
        print(f1_score_features)
        best_features.extend(X_train.columns)
        print(best_features)

        while np.sum(features_mask) > 0:
            for i in range(X_train.shape[1]):
                if features_mask[i]:
                    features_mask[i] = False
                    print(features_mask)
                    X_train_subset = X_train.loc[:, features_mask]
                    X_test_subset = X_test.loc[:, features_mask]
                    self.gbc.fit(X_train_subset, Y_train)
                    preds = self.gbc.predict(X_test_subset)
                    f1_score_subset = round(f1_score(Y_test, preds, average='weighted'), 3)
                    scores.append((f1_score_subset, original_indices[i]))
                    features_mask[i] = True
            print(scores)
            print(len(scores))
            worst_index = np.argmin([score[0] for score in scores])
            best_index = np.argmax([score[0] for score in scores])
            print("Worst index:", worst_index)
            print("Best index:", best_index)
            best_score = scores[best_index][0]
            if best_score >= f1_score_features:
                features_mask[scores[best_index][1]] = False
                print("\nNew Features Mask:\n", features_mask)
                print("Worst Feature Index:", worst_index)
                worst_feature = best_features[worst_index]
                print("Worst Feature:", worst_feature)
                best_features.remove(worst_feature)
                print("New Best Features:", best_features)
                X_train_best = X_train.loc[:, features_mask]
                X_test_best = X_test.loc[:, features_mask]
                self.gbc.fit(X_train_best, Y_train)
                preds = self.gbc.predict(X_test_best)
                f1_score_best = round(f1_score(Y_test, preds, average='weighted'), 3)
                f1_score_features = f1_score_best
                print("F1 Score with Best Features:", f1_score_best)
                scores.clear()
            else:
                print("\nNo improvement between previous score!\n")
                print("\nBest Features:", best_features, "\n")
                break

    def rfe_method(self) -> None:
        pass

    def initialize_population(self, population_size, num_features) -> None:
        population = np.random.randint(2, size=(population_size, num_features))
        print("Population generated!\n")
        return population

    def calculate_fitness(self, features, X_train, y_train, X_test, Y_test) -> None:
        feature_mask = features.astype(bool)
        X_train_subset = X_train.loc[:, feature_mask]
        X_test_subset = X_test.loc[:, feature_mask]
        self.gbc.fit(X_train_subset, y_train)
        preds = self.gbc.predict(X_test_subset)
        f1_score_subset = round(f1_score(Y_test, preds, average='weighted'), 3)
        # Update best individual if current individual is better
        global best_individual, best_fitness_score
        if f1_score_subset > best_fitness_score:
            best_fitness_score = f1_score_subset
            best_individual = features.copy()
        return f1_score_subset

    def calculate_individual_probabilities(self, individuals, fitness_scores_list, fitness_scores_sum) -> None:
        for i in range(len(fitness_scores_list)):
            probability = fitness_scores_list[i]/fitness_scores_sum
            individuals[i] = probability
        return individuals

    def selection_roulette_ga(self, population, population_size, output_prob) -> None:
        half_population_size = int(population_size/2)
        indices = list(output_prob.keys())
        selected_indices = np.random.choice(indices, half_population_size, p=list(output_prob.values()))
        selected_population = population[selected_indices]
        print(selected_population)
        return selected_population

    def selection_tournament_ga(self, population, tournament_size, fitness_score_dict) -> None:
        num_of_tournaments = len(population)//tournament_size
        print("Number of tournament:", num_of_tournaments)
        selected_population = []
        list_of_winners = []
        for _ in range(num_of_tournaments):
            indices = list(fitness_score_dict.keys())
            selected_indices = np.random.choice(indices, tournament_size, replace=False)
            print("Selected indices is: ",selected_indices)
            tournament_dict = {k: fitness_score_dict[k] for k in selected_indices}
            print(tournament_dict)
            winner = max(tournament_dict, key=lambda k: tournament_dict[k])
            correct_index_for_numpy = winner - 1
            list_of_winners.append(correct_index_for_numpy)
        print(population)
        selected_population = population[list_of_winners]
        print(selected_population)
        return selected_population
    
    def uniform_crossover(self, selected_population, population_size) -> None:
        next_gen = []
        i = population_size

        for _ in tqdm(range(i), desc='Crossover progress', colour = "green", leave=True):
            random_row_indices = np.random.choice(selected_population.shape[0], size=2, replace=False)
            random_parents = selected_population[random_row_indices, :]
            offspring = np.zeros((1, random_parents.shape[1]))

        for j in range(random_parents.shape[1]):
            offspring[:, j] = np.random.choice(random_parents[:, j], size=1)

        for k in range(offspring.shape[0]):
            next_gen.append(offspring[k])
        next_gen = np.array(next_gen, dtype=int)

        return next_gen
    
    def mutation(self, next_gen, selected_population) -> None:
        for i in range(next_gen.shape[0]):
           random_gene_from_individual = np.random.choice(selected_population.shape[1], size=1)
           if next_gen[i][random_gene_from_individual] == 1:
              next_gen[i][random_gene_from_individual] = 0
           else:
              next_gen[i][random_gene_from_individual] = 1
        # print("NEXT_GENERATION MUTATED:\n", next_gen)
        return next_gen

    def get_best_score(self, X_train):
        print("Best individual:", best_individual)
        print("Best fitness score:", best_fitness_score)
        mask = best_individual.astype(bool)
        final = X_train.loc[:, mask]
        print(final.columns)

    # Powinienem dac mozliwosc wyboru selekcji (jakas flaga)
    def run_genethic_algo(self, population_size, num_of_generations, features, filepath, target_col):
        global best_fitness_score
        global best_individual
        best_individual = None
        best_fitness_score = 0
         
        data = self.prepare_data(filepath=filepath)
    
        X, Y, X_train, X_test, Y_train, Y_test = self.split_data(data=data, target_col=target_col)
    
        population = self.initialize_population(population_size, features)
    
        for i in range (num_of_generations):
            print("Fitness calculation !")
            fitness_scores_list = [self.calculate_fitness(features, X_train, Y_train, X_test, Y_test) for features in tqdm(population, desc='Fitness Score Calculation Progress', colour = "green", leave=True)]
            fitness_scores_dict = {i + 1: score for i, score in enumerate(fitness_scores_list)}
            print("\nNajlepszy wynik dla generacji nr :", i+1 , "to :", max(fitness_scores_list))
            fitness_scores_sum = sum(fitness_scores_list)
            individ = {}
            output_prob = self.calculate_individual_probabilities(individ, fitness_scores_list, fitness_scores_sum)
            #selected_population = selection_roulette(population , population_size, output_prob)
            selected_population = self.selection_tournament_ga(population, 3, fitness_scores_dict)
            next_gen = self.uniform_crossover(selected_population, population_size)
            next_gen_mutated = self.mutation(next_gen, selected_population)
            population = next_gen_mutated
            print("Najlepszy fitness score", best_fitness_score)
        self.get_best_score(X_train)

    def generate_pheromone_array(self, num_features, value) -> None:
        pheromone_array = np.full(num_features, value)
        return pheromone_array
    
    def generate_ant_population(self, ants_colony_size, num_features) -> None:
        ants = np.zeros(shape=(ants_colony_size, num_features), dtype=int)
        for ant in ants:
            random_node_index = np.random.randint(0, num_features)
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

    def calcualate_ants_fitness(self, features, X_train, X_test, Y_train, Y_test) -> None:
        global best_individual, best_fitness_score
        feature_mask = features.astype(bool)
        X_train_subset = X_train.loc[:, feature_mask]
        X_test_subset = X_test.loc[:, feature_mask]
        self.gbc.fit(X_train_subset, Y_train)
        preds = self.gbc.predict(X_test_subset)
        f1_score_subset = round(f1_score(Y_test, preds, average='weighted'), 3)
        if f1_score_subset > best_fitness_score:
            best_fitness_score = f1_score_subset
            best_individual = features.copy()
        return f1_score_subset

    def update_pheromones(self, pheromone_array, ants_population, quality_measure, evaporation_rate=0.1) -> None:
        pheromone_array *= (1 - evaporation_rate)
        for ant in ants_population:
            for j in range(len(pheromone_array)):
                if ant[j] == 1:
                    # Update pheromone levels based on the quality measure
                    pheromone_array[j] += quality_measure[j]
        return pheromone_array

    def aco_run(self, X_train, X_test, Y_train, Y_test, base_pheromone_value, ants_colony_size, ant_generations, ants_moves, alpha, beta) -> None:
        mutual_info = mutual_info_classif(X_train, Y_train)
        weights = np.array(mutual_info)
        pheromone_array = self.generate_pheromone_array(len(mutual_info), base_pheromone_value)
        print("\nWagi:\n",weights)
        print("\nInitial pheromone_array: \n", pheromone_array)

        for _ in range(ant_generations):
            ants = self.generate_ant_population(ants_colony_size, len(mutual_info))
            for _ in range(ants_moves): 
                ants_population_after_move = self.ants_make_move(ants, pheromone_array=pheromone_array, weights=weights, alpha=alpha, beta=beta)
                fitness_scores_ants = [self.calculate_ants_fitness(features, X_train, X_test, Y_train, Y_test) for features in tqdm(ants_population_after_move, desc='Fitness Score Calculation Progress', colour = "green", leave=True)]
                print(max(fitness_scores_ants))
                print(fitness_scores_ants)
                updated_pheromones = self.update_pheromones(pheromone_array, ants_population_after_move, fitness_scores_ants)
                print("\nUpdated Pheromone Levels:\n", updated_pheromones)
                print("\nNew ants population\n:", ants_population_after_move)

        print(np.argmax(fitness_scores_ants))
        print(best_fitness_score)
