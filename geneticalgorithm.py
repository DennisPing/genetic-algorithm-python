import numpy as np
from random import randint, random, choice
from copy import deepcopy
import math

class Individual:
    """
    An individual is composed of a list of integers.
    The fitness of an individual is the sum of all squared integers.
    """
    def __init__(self, genome, target):
        self.genome = genome
        self.fitness = 0
        self.calcFitness(target)
    
    def __str__(self):
        return f"Genome: {self.genome}\nFitness Score: {self.fitness}"
    
    def calcFitness(self, target):
        fitness = 0
        for each in self.genome:
            fitness += each ** 2
        if fitness > target:
            self.fitness = 0
        else:
            self.fitness = round((fitness / target), 2)

class GeneticAlgorithm:
    """
    An algorithm that finds the best solution to a goal by simulating genetic evolution.
    """

    def __init__(self, genomeLength, popCount, target, low=0, high=5):
        """
        :param genomeLength: How many integers in an individual (this is N)
        :param popCount: How many individuals in a population
        :param target: The target integer (this is X)
        :param low: The minimum integer in the population, default is 0
        :param high: The maximum integer in the population, default is 5
        """
        if genomeLength < 2:
            raise ValueError(f"The minimum genome length is 2.")
        if popCount < 2:
            raise ValueError(f"The minimum population count is 2.")
        if low < 0:
            raise ValueError(f"The minimum possible integer cannot be negative.")
        self.genomeLength = genomeLength
        self.popCount = popCount
        self.target = target
        self.low = low
        self.high = high

    def _generateIndividual(self):
        """
        A helper function to generate an individual.
        :return: A list of random integers (repeats allowed) which represents an individual
        """
        genome = []
        for i in range(self.genomeLength):
            genome.append(randint(self.low, self.high))
        return Individual(genome, self.target)

    def generatePopulation(self):
        """
        Generate a population of individuals. Also checks if an entire population is unfit.
        :return: A list of individuals with which represents an entire population
        """
        population = []

        # Kill of 1 individual if population is odd. This indiv will never have offspring anyway.
        if not self.popCount % 2 == 0:
            self.popCount -= 1

        for i in range(self.popCount):
            population.append(self._generateIndividual())

        # Check edge case where all individuals are unfit. Replace population if all are unfit.
        fitnessScores = []
        while True:
            for individual in population:
                fitnessScores.append(individual.fitness)
            array = np.asarray(fitnessScores)
            if np.sum(array) > 0:
                break # Good population set
            else:
                population = self.generatePopulation() # Replace this bad population set
        population = sorted(population, key=lambda indiv: indiv.fitness)
        return population
    
    def selection(self, population):
        """
        First do elitism selection and trim to make population count even.
        Then do tournament style selection. All pairs of individuals must go through battle.
        :param population: A list of individuals in a population.
        :return: A list of parents who survived the tournament. Half the population.
        """
        parents = []
        numElites = 0
        # The best 10% is automatically carried onto next generation
        if self.popCount >= 10: # Avoid carrying over elites when population is under 10
            numElites = math.floor(self.popCount * 0.1)
            if not (self.popCount - numElites) % 2 == 0:
                numElites += 1
        
        # For 'n' number of elites, remove them from the future tournament
        for _ in range(numElites):
            elite = population.pop()
            parents.append(elite)
        
        # Since these elites took up spots in the parent list, remove an equal number of population
        for _ in range(numElites):
            population.pop(0)

        # While there are still individuals alive, battle the first and last ones
        while population:
            pleb1 = population.pop()
            pleb2 = population.pop(0)
            if pleb1.fitness >= pleb2.fitness:
                parents.append(pleb1)
            else:
                parents.append(pleb2)
        
        parents = sorted(parents, key=lambda indiv: indiv.fitness)
        return parents

    def crossover(self, parents):
        """
        Crossover every pair of parents to create two children per pair.
        :param parents: A list of winning parents.
        :return: A list of winning parents and a list of new children.
        """
        children = []
        recovered = [] # Recover the parents that you pop out
        alpha = deepcopy(parents[-1]) # Store this best parent in case of odd number of parents
        odd = False
        while parents:
            mom = parents.pop(0)
            recovered.append(mom)
            if parents:
                dad = parents.pop()
                recovered.append(dad)
            else:
                dad = alpha
                odd = True

            # Apply a bell curve bias to the cut index
            dice1 = randint(0, self.genomeLength)
            dice2 = randint(0, self.genomeLength)
            idx = round((dice1 + dice2) / 2)

            # Create two children at the cut idx from mom and dad
            egg1 = mom.genome[0:idx] + dad.genome[idx:]
            child1 = Individual(egg1, self.target)
            egg2 = dad.genome[0:idx] + mom.genome[idx:]
            child2 = Individual(egg2, self.target)

            # Handle both cases when you have even or odd number parents
            if odd is False:
                children.append(child1)
                children.append(child2)
            else:
                child = max(child1, child2, key=lambda indiv: indiv.fitness)
                children.append(child)
        return recovered, children

    def mutation(self, children, mutationRate=0.05):
        """
        Each genome in each children has a chance to mutate.
        :param children: A list of children
        :param mutationRate: The mutation rate, default is 5%
        """
        mutatedChildren = []

        for child in children:
            # Need to clone our child because we cannot mutate lists while iterating
            clone = deepcopy(child)
            for i, gene in enumerate(child.genome):
                if random() < mutationRate:
                    mutGene = self._getNewGene(gene, self.low, self.high)
                    clone.genome[i] = mutGene
            clone.calcFitness(self.target) # Remember to recalculate the fitness
            mutatedChildren.append(clone)
        return mutatedChildren

    def _getNewGene(self, exclude, low, high):
        """
        :param exclude: The original gene to exclude.
        :param low: The minimum gene value
        :param high: The maximum gene value
        :return: A new random gene that is not the original gene
        """
        options = list(range(low, high + 1))
        return choice([g for g in options if g != exclude])

    def cycleOfLife(self, parents, mutatedChildren):
        """
        Create a new population of winning parents + children. Thus removing the losing parents.
        :param parents: The winning parents
        :param mutatedChildren: The mutated children
        :return: A new population of winning parents + children.
        """
        population = parents + mutatedChildren
        population = sorted(population, key=lambda indiv: indiv.fitness)
        return population
    
    def getBest(self, population):
        """
        The population is already sorted so just look at the last element.
        :param population: The population
        :return: The best individual in the population
        """
        best = population[-1]
        return best

class Prompt:
    """
    A class to ask the user for valid input. Also supports quit functionality.
    """
    def __init__(self):
        pass

    def userPrompt(self):
        prompts = []
        prompts.append("Enter a genome length: ")
        prompts.append("Enter a population count: ")
        prompts.append("Enter a goal number: ")
        prompts.append("Enter the lowest gene number allowed: ")
        prompts.append("Enter the highest gene number allowed: ")

        answers = []
        for each in prompts:
            ans = self.getValidNumber(each)
            if ans == 'quit':
                return
            answers.append(ans)
        return answers[0], answers[1], answers[2], answers[3], answers[4]
    
    def getValidNumber(self, message):
        while True:
            try:
                userInput = input(message)
                if userInput.lower() == 'q' or userInput.lower() == 'quit':
                    return 'quit'
                else:
                    userInput = int(userInput)
            except ValueError:
                print("Please enter a number, or type (quit/q) to quit.")
                continue
            else:
                break
        return userInput
        
def main():
    # Recommended parameters: genomeLength=5, popCount=10, target=105, low=0, high=5
    prompt = Prompt()
    answers = prompt.userPrompt()
    if answers:
        genomeLength, popCount, target, low, high = answers
        evolution = GeneticAlgorithm(genomeLength, popCount, target, low, high)
        population = evolution.generatePopulation()
    else:
        return

    best = None
    generationLimit = 200
    count = 0
    for gen in range(generationLimit):
        count += 1
        # Check if you've already reached the target
        best = evolution.getBest(population)
        if best.fitness == 1.0:
            print(f"Found a perfect individual after '{count}' generations:")
            print(best)
            break
        if gen == generationLimit - 1:
            print(f"This is the closest I can get after '{count}' generations:")
            print(best)
            break
        parents = evolution.selection(population)
        winningParents, children = evolution.crossover(parents)
        mutatedChildren = evolution.mutation(children)
        population = evolution.cycleOfLife(winningParents, mutatedChildren)
    
if __name__ == "__main__":
    main()
