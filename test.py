import requests
import lkh

# problem_str = requests.get('http://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/P/P-n16-k8.vrp').text
file_path = 'P-n16-k8.vrp'
with open(file_path, 'r') as file:
    problem_str = file.read()
problem = lkh.LKHProblem.parse(problem_str)

solver_path = 'LKH-3.exe'
a = lkh.solve(solver_path, problem=problem, max_trials=10000, runs=1)

print(a)