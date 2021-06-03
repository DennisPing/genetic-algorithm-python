# Genetic Algorithm

**Dennis Ping**
**Homework 2**
**June 4, 2021**

### Purpose

Implement a genetic algorithm for creating a list of 'N' numbers that equals 'X' when squared and summed together.

### Python and Pip requirements

- Python 3
- numpy

### How to run

- `python geneticalgorithm.py`

### Example output

Finding the best outcome where this is no perfect outcome:
*Goal number is 105*

```rtf
This is the closest I can get after '200' generations:
Genome: [5, 5, 2, 5, 5]
Fitness Score: 0.99
```

Finding the best outcome where there is a perfect outcome:
*Goal number is 100*

```rtf
Found a perfect individual after '4' generations:
Genome: [4, 5, 3, 5, 5]
Fitness Score: 1.0
```

### Input criteria

- The genome length (number of integers in an individual) greater than or equal to 2. Otherwise you cannot split the genome.
- The population count must be greater than or equal to 2. Otherwise you cannot do gene crossover.
