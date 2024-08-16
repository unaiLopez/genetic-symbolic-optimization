class MetaGeneticSymbolicRegressor:
    def __init__(self, mode: str):
        self.mode = mode

    def find_best_prior_symbol_subsets(self):
        # combinations of families trigonometric, arithmetic, exponential, statistics, etc

        # define a scoring system based on score (the smaller the better)
        # define a scoring system based on number of symbols in subset (the smaller the better)
        # define complexity of max tree depth
        # run N trials in order to ensure it is the best and reduce uncertainty
        
        # the objective of this function is to constraint the complexity of the search space doing a first quick step
        pass

    def advanced_search(self):
        pass