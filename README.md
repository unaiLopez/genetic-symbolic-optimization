Multiples poblaciones a la vez ¿Con crossover entre mutaciones?
Implementar más mutaciones (if mutacion then implementar aleatoriamente alguna de todas las opciones)
Hacer crossover y mutaciones más robustas evitando que se supere el max depth entre otras cosas...
Revisar el perform selection y el tournament

Arreglo de el best siempre tiene que ser el que menos loss tiene y ademas menos complexity
Convertir formulas en numpy a sympy
Añadir constantes con optimizacion de constantes (scipy)
Implementar losses (mae, rmse, mse, etc)

Añadir features de visualizacion de optimizacion
    - Eje X Loss y eje Y complexity
    - 1 Dataframe por cada generacion
    - Tener 1 dataframe con el mejor invididuo por complexity

Implementar benchmarks.py -> Con esto podremos ir evaluando si los cambios ayudan a mejorar o no (mirar benchmarks de PySr)