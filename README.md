# Backlog

- ~~Hacer crossover y mutaciones más robustas evitando que se supere el max depth entre otras cosas...~~ -> DONE
- ~~Implementar losses (mae, rmse, mse, etc)~~ -> DONE
- ~~Añadir features de visualización de optimización~~ -> DONE
  - ~~Eje X Loss y eje Y complexity~~ -> DONE
  - ~~1 Dataframe por cada generación~~ -> DONE
  - ~~Tener 1 dataframe con el mejor individuo por complexity~~ -> DONE
  - ~~Visualizar evolución (best_loss, worst_loss and mean_loss)~~ -> DONE

- Revisar el perform selection y el tournament
- Múltiples poblaciones a la vez ¿Con crossover entre poblaciones?
- Implementar más mutaciones (if mutación then implementar aleatoriamente alguna de todas las opciones)
- Arreglo de el best siempre tiene que ser el que menos loss tiene y además menos complexity
- Convertir fórmulas en numpy a sympy
- Añadir constantes con optimización de constantes (scipy)
- Implementar operadores min, max, mean (esto podría ser interesante para el ML)
- Implementar benchmarks.py
  - Con esto podremos ir evaluando si los cambios ayudan a mejorar o no (mirar benchmarks de PySr)
- Probar métrica BIC, AIC o Weighted_IC:
  - BIC = n⋅ln(σ^2) + k⋅ln(n)
  - AIC = n⋅ln(RSS / n​) + 2⋅k
  - Weighted_IC = α⋅AIC+(1 − α)⋅BIC

- Divide and conquer approach
  - Generate a big subset of subexpressions
  - After generating a big subset of subexpressions (individuals), go to second stage. Search for combinations between them (Genetic Algorithms)
