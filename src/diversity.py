from typing import List, Any

def unique_individuals_ratio(individuals: List[Any]) -> float:
    equations = [individual[5] for individual in individuals]
    
    return len(set(equations)) / len(equations)

