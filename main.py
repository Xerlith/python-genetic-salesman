import numpy as np
import random, operator
import pandas as pd

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
    
    def routeDistance(self):
        if self.distance == 0:
            distance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                # Accounts for starting and ending in the same place
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                distance += fromCity.distance(toCity)

            self.distance = distance
        return self.distance
    
    
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

# setup functions
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []

    for _ in range(0, popSize):
        population.append(createRoute(cityList))

    return population

#selection functions
def rankRoutes(population):
    fitnessResults = {}

    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()

    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, survivorSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, survivorSize):
        selectionResults.append(popRanked[i][0])

    for i in range(0, len(popRanked) - survivorSize):
        pick = 100 * random.random()

        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break

    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breedPopulation(matingpool, survivorSize):
    children = []
    length = len(matingpool) - survivorSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,survivorSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

def mutatePopulation(population, mutationRate):
    newPopulation = []
    
    for individual in range(0, len(population)):
        newIndividual = mutate(population[individual], mutationRate)
        newPopulation.append(newIndividual)
    return newPopulation
    
def advanceGeneration(currentGen, survivorSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, survivorSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, survivorSize)
    advanceGeneration = mutatePopulation(children, mutationRate)
    return advanceGeneration

# main function
def geneticAlgorithm(populationSize, survivorSize, mutationRate, generations):
    cityList = []

    for _ in range(0,25):
        cityList.append(City(x=int(random.random() * 500), y=int(random.random() * 500)))

    population = initialPopulation(populationSize, cityList)
    print("Dystans początkowy: " + str(1 / rankRoutes(population)[0][1]))
    
    for _ in range(0, generations):
        population = advanceGeneration(population, survivorSize, mutationRate)
    
    print("Dystans końcowy: " + str(1 / rankRoutes(population)[0][1]))
    bestRouteIndex = rankRoutes(population)[0][0]
    bestRoute = population[bestRouteIndex]
    return bestRoute



geneticAlgorithm(populationSize=100, survivorSize=20, mutationRate=0.01, generations=1000)
