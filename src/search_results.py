import os
import platform

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from typing import List, Any

pd.set_option('display.max_columns', None)

class SearchResults:
    def __init__(self):
        self.best_by_loss_complexity_per_epoch = list()
        self.best_by_loss_complexity = None
        self.df_summary_statistics = pd.DataFrame()

    def add_best_individuals_by_loss_and_complexity(self, individuals: List[Any], generation: int) -> None:
        epoch_results = list()
        for individual in individuals:
            epoch_results.append({
                "best": "",
                "score": individual[2],
                "loss": individual[1],
                "complexity": individual[7],
                "tree_depth": individual[4],
                "equation": individual[5],
                "generation": generation,
                "current_generation": generation
            })
            
        df_epoch_results = pd.DataFrame(epoch_results)
        df_epoch_results.sort_values(by=["complexity", "loss"], ascending=True, inplace=True)
        df_epoch_results = df_epoch_results.groupby("complexity").first().reset_index()
        df_epoch_results = df_epoch_results[["best", "score", "loss", "complexity", "tree_depth", "equation", "generation"]]

        if self.best_by_loss_complexity is None:
            self.best_by_loss_complexity = df_epoch_results.copy()
        else:
            self.best_by_loss_complexity = pd.concat([self.best_by_loss_complexity, df_epoch_results], axis=0)
            self.best_by_loss_complexity.sort_values(by=["complexity", "loss"], ascending=True, inplace=True)
            self.best_by_loss_complexity = self.best_by_loss_complexity.groupby("complexity").first().reset_index()
            self.best_by_loss_complexity = self.best_by_loss_complexity[["best", "score", "loss", "complexity", "tree_depth", "equation", "generation"]]
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

    def extract_summary_statistics_from_individuals(self, individuals: List[Any], generation: int) -> None:
        scores = [individual[2] for individual in individuals if individual[2]]
        df_summary_stats_by_generation = pd.DataFrame({
            "generation": [generation],
            "max": [np.max(scores)],
            "min": [np.min(scores)],
            "std": [np.std(scores)],
            "var": [np.var(scores)],
            "mean": [np.mean(scores)]
        })
        if self.df_summary_statistics is None:
            self.df_summary_statistics = df_summary_stats_by_generation.copy()
        else:
            self.df_summary_statistics = pd.concat([self.df_summary_statistics, df_summary_stats_by_generation], axis=0)

    
    def plot_evolution(self):
        # Ensure the DataFrame has data for multiple generations
        if len(self.df_summary_statistics) == 0:
            raise ValueError("DataFrame is empty. Please provide data with multiple generations.")

        # Initialize the figure
        fig = go.Figure()

        # Add traces for each statistic
        for stat in ['max', 'min', 'mean']:
            fig.add_trace(go.Scatter(
                x=self.df_summary_statistics['generation'],
                y=self.df_summary_statistics[stat],
                mode='lines+markers',
                name=stat.capitalize(),
                hoverinfo='x+y+name',
                line=dict(width=2),
                marker=dict(size=6)
            ))

        # Update layout for better aesthetics
        fig.update_layout(
            title='Summary Statistics of Genetic Algorithm over Generations',
            xaxis_title='Generation',
            yaxis_title='Value',
            legend_title='Statistic',
            template='plotly_white',  # Clean background
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='x unified'  # Hover over one point to show all statistics
        )

        # Show plot
        fig.show()

    def plot_evolution_per_complexity(self):
        # Combine data into a single DataFrame
        best_per_epoch = pd.concat(self.best_by_loss_complexity_per_epoch)

        # Initialize the figure
        fig = go.Figure()

        # Iterate through each complexity level and add traces
        for complexity, group_df in best_per_epoch.groupby('complexity'):
            fig.add_trace(go.Scatter(
                x=group_df['generation'],  # Assuming 'generation' is a column in the DataFrame
                y=group_df['score'],
                mode='lines',
                name=f'Complexity {complexity}',
                hoverinfo='x+y+name',  # Show x, y, and trace name on hover
                line=dict(width=2)
            ))

        # Update layout for better aesthetics and logarithmic Y-axis
        fig.update_layout(
            title='Score vs Generation for Different Complexities',
            xaxis_title='Generation',
            yaxis_title='Score',
            legend_title='Complexity',
            legend=dict(x=0.85, y=0.95),
            template='plotly_white',  # Clean background
            margin=dict(l=50, r=50, t=50, b=50)
        )
        # Show plot
        fig.show()