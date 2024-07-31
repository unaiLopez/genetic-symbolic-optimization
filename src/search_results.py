import os
import platform
import pandas as pd

from typing import List
from src.binary_tree import BinaryTree

class SearchResults:
    def __init__(self):
        self.best_by_loss_complexity_per_epoch = list()
        self.best_by_loss_complexity = None

    def add_best_individuals_by_loss_and_complexity(self, individuals: List[BinaryTree], generation: int) -> None:
        epoch_results = list()
        for individual in individuals:
            epoch_results.append({
                "best": "",
                "generation": generation,
                "loss": individual.loss,
                "complexity": individual.complexity,
                "tree_depth": individual.depth,
                "equation": individual.equation,
                "current_generation": generation
            })
            
        df_epoch_results = pd.DataFrame(epoch_results)
        df_epoch_results.sort_values(by=["complexity", "loss"], ascending=True, inplace=True)
        df_epoch_results = df_epoch_results.groupby("complexity").first().reset_index()
        df_epoch_results = df_epoch_results[["best", "generation", "loss", "complexity", "tree_depth", "equation"]]

        if self.best_by_loss_complexity is None:
            self.best_by_loss_complexity = df_epoch_results.copy()
        else:
            self.best_by_loss_complexity = pd.concat([self.best_by_loss_complexity, df_epoch_results], axis=0)
            self.best_by_loss_complexity.sort_values(by=["complexity", "loss"], ascending=True, inplace=True)
            self.best_by_loss_complexity = self.best_by_loss_complexity.groupby("complexity").first().reset_index()
            self.best_by_loss_complexity = self.best_by_loss_complexity[["best", "generation", "loss", "complexity", "tree_depth", "equation"]]
            self.best_by_loss_complexity.loc[:, "best"] = ""
            if len(self.best_by_loss_complexity[self.best_by_loss_complexity["loss"] == self.best_by_loss_complexity["loss"].min()]) > 1: # ESTO NO SIEMPRE FUNCIONA PORQUE HAY MUCHOS DECIMALES Y NO SON IGUALES
                df_min_loss = self.best_by_loss_complexity[self.best_by_loss_complexity["loss"] == self.best_by_loss_complexity["loss"].min()]
                df_min_loss = df_min_loss[df_min_loss["complexity"] == df_min_loss["complexity"].min()]
                min_index = df_min_loss.index
            else:
                min_index = self.best_by_loss_complexity["loss"].idxmin()
            self.best_by_loss_complexity.loc[min_index, "best"] = ">>>"
            self.best_by_loss_complexity.loc[:, "current_generation"] = generation
        
        self.best_by_loss_complexity_per_epoch.append(self.best_by_loss_complexity)

    def visualize_best_in_generation(self):
        if platform.system() == "Windows":
            os.system("cls")
        else:
            os.system("clear")
        print(self.best_by_loss_complexity)
    
    def plot_evolution_per_complexity(self):
        import matplotlib.pyplot as plt

        best_per_epoch = pd.concat(self.best_by_loss_complexity_per_epoch)
        plt.figure(figsize=(20, 10))
        for complexity, group_df in best_per_epoch.groupby('complexity'):
            plt.plot(group_df['loss'].values, label=f'Complexity {complexity}')

        # Add labels and legend
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.title('Loss vs Generation for Different Complexities')
        plt.legend(title='Complexity')
        plt.grid(True)

        # Show plot
        plt.show()

                    
                    

