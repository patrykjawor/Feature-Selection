import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, mean_squared_error, confusion_matrix
from numba import jit

# Genetic Alghorithm
population_size = 100
features = 30
num_of_generations = 2

# FUNKCJA DO PRZYGOTOWANIA DANYCH (ZEBY BYLO PEWNIEJ POWINIENEM SPRAWDZAC CZY DANE MAJA OUTLIERY I DOBIERAC WTEDY ALBO MEAN() ALBO MEDIAN())
# JESZCZE PRZEMYSLEC ROZNE PRZYPADKI!!!!
def prepare_data(path):
    data = pd.read_csv(filepath_or_buffer=path)
    label_encoder = preprocessing.LabelEncoder()
    nan_count = data.isnull().sum()
    print(nan_count)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')] 
    data = data.apply(lambda col: col.fillna("NO INFO") if is_object_dtype(col) else col)
    data = data.apply(lambda col: label_encoder.fit_transform(col) if is_object_dtype(col) else col)
    data = data.apply(lambda col: col.fillna(col.median()) if is_numeric_dtype(col) else col)
    nan_count = data.isnull().sum()
    print(nan_count)
    return data

# STEP 1: Initialization operator
def initialize_population(population_size, num_features):
    population = np.random.randint(2, size=(population_size, num_features))
    print("Population generated!\n")
    return population

# STEP 2: Fitness Function
def fitness(features, classifier, X_train, y_train, X_test, y_test):
   feature_mask = features.astype(bool)
   X_train_subset = X_train.loc[:, feature_mask]
   X_test_subset = X_test.loc[:, feature_mask]
   classifier.fit(X_train_subset, y_train)
   preds = classifier.predict(X_test_subset)
   f1_score_subset = round(f1_score(y_test, preds, average='weighted'), 3)
# Update best individual if current individual is better
   global best_individual, best_fitness_score
   if f1_score_subset > best_fitness_score:
       best_fitness_score = f1_score_subset
       best_individual = features.copy()
   return f1_score_subset

# STEP 3: Selection
def calc_individual_probabilities(individuals, fitness_scores_list, fitness_scores_sum):
    for i in range(len(fitness_scores_list)):
        probability = fitness_scores_list[i]/fitness_scores_sum
        individuals[i] = probability
    return individuals

def selection_roulette(population, population_size, output_prob):
    half_population_size = int(population_size/2)
    indices = list(output_prob.keys())
    selected_indices = np.random.choice(indices, half_population_size, p=list(output_prob.values()))
    selected_population = population[selected_indices]
    print(selected_population)
    return selected_population

def selection_tournament(population, tournament_size, fitness_score_dict):
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

#STEP 4: Crossover
def uniform_crossover(selected_population, population_size):
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

#Step 5: Mutation
def mutation(next_gen, selected_population):
    for i in range(next_gen.shape[0]):
       random_gene_from_individual = np.random.choice(selected_population.shape[1], size=1)
       if next_gen[i][random_gene_from_individual] == 1:
          next_gen[i][random_gene_from_individual] = 0
       else:
          next_gen[i][random_gene_from_individual] = 1
    # print("NEXT_GENERATION MUTATED:\n", next_gen)
    return next_gen


def run_genethic_algo(population_size, num_of_generations, features):
    global best_fitness_score
    global best_individual
    best_individual = None
    best_fitness_score = 0
     
    data = prepare_data("/Users/patrykjaworski/Documents/Projekty/Feature-Selection/TESTY/Old/data.csv")
    X = data.drop(['id','diagnosis'] , axis=1)
    y = data['diagnosis']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.3,
                                                   random_state=42)
    # Initialize classifier
    classifier = GradientBoostingClassifier(random_state=42)

    population = initialize_population(population_size, features)

    for i in range (num_of_generations):
        print("Fitness calculation !")
        fitness_scores_list = [fitness(features, classifier, X_train, y_train, X_test, y_test) for features in tqdm(population, desc='Fitness Score Calculation Progress', colour = "green", leave=True)]
        fitness_scores_dict = {i + 1: score for i, score in enumerate(fitness_scores_list)}
        print("\nNajlepszy wynik dla generacji nr :", i+1 , "to :", max(fitness_scores_list))
        fitness_scores_sum = sum(fitness_scores_list)
        individ = {}
        output_prob = calc_individual_probabilities(individ, fitness_scores_list, fitness_scores_sum)
        #selected_population = selection_roulette(population , population_size, output_prob)
        selected_population = selection_tournament(population, 3, fitness_scores_dict)
        next_gen = uniform_crossover(selected_population, population_size)
        next_gen_mutated = mutation(next_gen, selected_population)
        population = next_gen_mutated
        print("Najlepszy fitness score", best_fitness_score)

    get_best_score(X_train)

def get_best_score(X_train):
    print("Best individual:", best_individual)
    print("Best fitness score:", best_fitness_score)
    mask = best_individual.astype(bool)
    final = X_train.loc[:, mask]
    print(final.columns)

run_genethic_algo(30,20,30)
