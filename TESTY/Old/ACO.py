import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, mean_squared_error
from itertools import accumulate
import operator

data = pd.read_csv("/Users/patrykjaworski/Documents/Projekty/Feature-Selection/TESTY/Old/data.csv")
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

#Dzielimy dane na uczące i testowe 
X = data.drop(['id','diagnosis'], axis=1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.3,
                                                   random_state=42)
best_individual = None
best_fitness_score = 0

# Initialize classifier
gbc = GradientBoostingClassifier(max_depth=5, random_state=42)  # Dla problemu klasyfikacji

# TEST
# Krok 1 Ranking parametrów
mutual_info = mutual_info_classif(X_train, y_train)


# KROK 2
# DEFINIUJE PHEROMONE ARRAY (TAKA SAMA WARTOSC)
def generate_pheromone_array(num_features, value):
    pheromone_array = np.full(num_features, value)
    return pheromone_array

# Krok 3
# Generacja początkowego ustawienia mrówek
def generate_ant_population(ants_colony_size, num_features):
    ants = np.zeros(shape=(ants_colony_size, num_features), dtype=int)
    for ant in ants:
        random_node_index = np.random.randint(0, num_features)
        ant[random_node_index] = 1
    print("Ants generated!\n")
    return ants

# KROK 4 KALKULACJA PRAWDOPODBIENSTWA DLA KANDYDATOW i SELEKCJA NOWEJ CECHY
def roulette_wheel_selection(probabilities, num_candidates):
    print("\nProblities:\n", probabilities)
    cumulative_probabilities = list(accumulate(probabilities, operator.add))
    print("\nCumulative probabilities: \n", cumulative_probabilities)
    selected_index = np.searchsorted(cumulative_probabilities, np.random.rand()) % num_candidates
    return selected_index

def ants_make_move(ants_population, pheromone_array, weights):
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
    
        selected_feature_index = roulette_wheel_selection(probabilities_for_each_candidate, len(probabilities_for_each_candidate))
        print("Selected Feature Index:", selected_feature_index)
        print()
        current_ant[selected_feature_index] = 1
        print("Ant selected:\n", current_ant)
        print()

    return ants_population

# Krok 5 SPRAWDZ FITNESS MRÓWEK
def calculate_ant_fitness(features):
   global best_individual, best_fitness_score
   feature_mask = features.astype(bool)
   X_train_subset = X_train.loc[:, feature_mask]
   X_test_subset = X_test.loc[:, feature_mask]
   gbc.fit(X_train_subset, y_train)
   preds = gbc.predict(X_test_subset)
   f1_score_subset = round(f1_score(y_test, preds, average='weighted'), 3)
   if f1_score_subset > best_fitness_score:
       best_fitness_score = f1_score_subset
       best_individual = features.copy()
   return f1_score_subset

# KROK 6 ZAKTUALIZUJ LISTE FEROMONÓW NA PODSTAWIE WYBORÓW MRÓWEK
def update_pheromones(pheromone_array, ants_population, quality_measure, evaporation_rate=0.1):
    pheromone_array *= (1 - evaporation_rate)
    for ant in ants_population:
        for j in range(len(pheromone_array)):
            if ant[j] == 1:
                # Update pheromone levels based on the quality measure
                pheromone_array[j] += quality_measure[j]
    return pheromone_array

alpha = 1.0
beta = 1.0

weights = np.array(mutual_info)
pheromone_array = generate_pheromone_array(30, 0.2)
print("\nWagi:\n",weights)
print("\nInitial pheromone_array: \n", pheromone_array)

ant_generations = 3
for j in range(ant_generations):
    ants = generate_ant_population(30, 30)
    for i in range(10): 
        ants_population_after_move = ants_make_move(ants, pheromone_array=pheromone_array, weights=weights)
        fitness_scores_ants = [calculate_ant_fitness(features) for features in tqdm(ants_population_after_move, desc='Fitness Score Calculation Progress', colour = "green", leave=True)]
        print(max(fitness_scores_ants))
        print(fitness_scores_ants)
        updated_pheromones = update_pheromones(pheromone_array, ants_population_after_move, fitness_scores_ants)
        print("\nUpdated Pheromone Levels:\n", updated_pheromones)
        print("\nNew ants population\n:", ants_population_after_move)

print(np.argmax(fitness_scores_ants))
print(best_fitness_score)
