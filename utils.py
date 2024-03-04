import numpy as np
import random
def get_concepts(text_path):
    with open(text_path, 'r') as file:
        return [concept.split('\n')[0] for concept in file.readlines()]
def label2concept(labels,concepts_all):
    concepts=[]
    # n=len(concepts_all)
    # covid_concept=[1.0]*n
    # non_concept=[0.0]*n
    n=len([concept for concept in concepts_all if 'NOT' not in concept])
    covid_concept=[1.0]*n+[0.0]*(len(concepts_all)-n)
    non_concept=[0.0]*n+[1.0]*(len(concepts_all)-n)
    for label in labels:
        if label==0:
            concepts.append(covid_concept)
        else:
            concepts.append(non_concept)
            
    return concepts
def labeltoconcepts(labels,concepts_all):
    concepts=[]
    
    # covid_concept=[1.0]*40+[0.0] * 40+[random.uniform(0.0, 1.0) for _ in range(80)]
    # covid_concept = [1.0 if val > 0.5 else 0.0 for val in covid_concept]
    # non_concept=[0.0]*40+[1.0] * 40+[random.uniform(0.0, 1.0) for _ in range(80)]
    # non_concept = [1.0 if val > 0.5 else 0.0 for val in non_concept]
    covid_concept=[1.0]*40+[0.0] * 40+[1.0]*80
    non_concept=[0.0]*40+[1.0] * 40+[1.0]*80
    for label in labels:
        if label==0:
            concepts.append(covid_concept)
        else:
            concepts.append(non_concept)
            
    return concepts