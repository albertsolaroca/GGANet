# BELOW CODEBLOCK IS DEPRECATED PLATYPUS CODE
# problem = Problem(1, 1, 9) # We have one pump and one objective function (energy). Not sure about the constraints other than the pump pattern values being 0 <= x <= 1
# problem.types[:] = Binary(24) # 24 hours in a day => 24 pump pattern multipliers
# problem.constraints[:] = ">=0" # Not sure about this
# problem.function = minimization_function # The function we want to minimize
# problem.directions[:] = Problem.MINIMIZE
# #%%
# algorithm = GeneticAlgorithm(problem, population_size=50)
# algorithm.run(100)
#
# non_dominated_solutions = nondominated(algorithm.result)
# #%%
# bitwise_solutions = []
#
# for solution in non_dominated_solutions:
#     bitwise_solutions.append([])
#     for i in range(len(solution.variables)):
#         for entry in solution.variables[i]:
#             if entry:
#                 bitwise_solutions[-1].append(1)
#             else:
#                 bitwise_solutions[-1].append(0)
#
#     print("Solution schedule", bitwise_solutions[-1],
#           "Total energy spent", solution.objectives)


# BELOW CODEBLOCK IS DEPRECATED PYMOO CODE WITH SINGLE OBJECTIVE OPTIMIZATION
# algorithm = GA(
#     pop_size=100,
#     sampling=BinaryRandomSampling(),
#     # crossover=TwoPointCrossover(), Apparently for a single pump there is no point to this? Check Asana
#     mutation=BitflipMutation(),
#     eliminate_duplicates=True)
#
# termination = get_termination("n_gen", 10)
#
# res = minimize(problem,
#                algorithm,
#                termination,
#                seed=12343,
#                verbose=True)