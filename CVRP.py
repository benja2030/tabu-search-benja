from itertools import combinations
import numpy as np

class Trucks:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity  # Capacidad del camion
        self.packages = 0  # Paquetes que lleva
        self.route = []  # Ruta con clientes

    def __repr__(self):
        return (
            f"Truck(id={self.id}, capacity={self.capacity}, packages={self.packages}, "
            f"route={self.route})"
        )


class Clients:
    def __init__(self, id, x, y, demand):
        self.id = id
        self.x = x  # Posicion del cliente
        self.y = y
        self.demand = demand  # Demanda del cliente

    def __repr__(self):
        return f"{self.id}"


def parse_vrp_file(file_path):
    clients = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        node_section = False
        demand_section = False
        depot_section = False
        
        for line in lines:
            if line.startswith("CAPACITY"):
                capacity = int(line.split()[-1])
            elif line.startswith("NODE_COORD_SECTION"):
                node_section = True
                demand_section = False
                depot_section = False
            elif line.startswith("DEMAND_SECTION"):
                node_section = False
                demand_section = True
                depot_section = False
            elif line.startswith("DEPOT_SECTION"):
                node_section = False
                demand_section = False
                depot_section = True
            elif line.startswith("EOF"):
                break
            elif node_section:
                parts = line.split()
                id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                clients.append(Clients(id=id, x=x, y=y, demand=0))
            elif demand_section:
                parts = line.split()
                id = int(parts[0])
                demand = int(parts[1])
                for client in clients:
                    if client.id == id:
                        client.demand = demand
                        break
            elif depot_section:
                depot_id = int(line.strip())
                if depot_id == -1:
                    continue
    return capacity, clients

# Crea una matriz con las distancias euclidiana de los nodos
# O(n^2)
def calculate_distance_matrix(clients):
    num_clients = len(clients)
    distance_matrix = np.zeros((num_clients, num_clients))  # Matriz vacia

    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                distance_matrix[i][j] = np.sqrt(
                    (clients[i].x - clients[j].x) ** 2
                    + (clients[i].y - clients[j].y) ** 2
                )  # Distancia euclidiana

    return distance_matrix

# Calcula la distancia entre el ultimo cliente de la ruta y algun otro cliente
# O(1)
def check_costs(truck, client, dist_matrix, clients):
    if len(truck.route) == 1:
        last_client = clients[0].id-1
    else:
        last_client = truck.route[-1].id-1
    travel_distance = dist_matrix[last_client][client.id-1]
    # Retorna la distancia hasta un cliente
    return travel_distance

# Crea la solucion inicial
# O(k*n^2)
def create_initial_solution(trucks, clients, dist_matrix):
    # Clientes no visitados
    unvisited_clients = clients[1:]
    # Se crean rutas para cada camion
    for truck in trucks:
        truck.route.append(clients[0])
        # Se agregan clientes mientras que queden clientes sin visitar
        while unvisited_clients:
            best_cost = float("inf")
            best_client = None
            for client in unvisited_clients:
                if truck.packages + client.demand <= truck.capacity:
                    current_cost = check_costs(truck, client, dist_matrix, clients)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_client = client
            if best_client is None:
                print(f"Can't add more clients {truck}")
                break
            truck.packages += best_client.demand
            truck.route.append(best_client)
            unvisited_clients.remove(best_client)
        truck.route.append(clients[0])

# Calcula la distancia total recorrida por una solucion
# O(k*n)
def total_route_cost(solution,dist_matrix):
    traveled_distance = 0
    for route in solution:
        for i in range(len(route) - 1):
            traveled_distance += dist_matrix[route[i].id-1][route[i + 1].id-1]
    return traveled_distance

# Revisa si una solucion es factible
# O(k*n)
def is_solution_feasible(solution, capacity):
    for route in solution:
        route_demand = sum(client.demand for client in route)
        if route_demand > capacity:
            return False
    return True

# Crea una lista de vecinos ordenados
# La creacion de vecinos se hace a travez del intercambio de nodos entre rutas
# 
def create_neighbourhood1(sol, dist_matrix, capacity):
    neighbours = []
    for combo in list(combinations(sol, 2)):
        for i in combo[0][:-1]:
            for j in combo[1][:-1]:                
                if i.id == 1 or j.id == 1:
                    continue
                temp = [list(item) for item in sol]
                c0 = list(combo[0])
                c1 = list(combo[1])
                index1 = temp.index(c0)
                index2 = temp.index(c1)
                c1.insert(c1.index(j),i)
                c0.insert(c0.index(i),j)
                c1.remove(j)
                c0.remove(i)
                temp[index1] = c0
                temp[index2] = c1
                if (is_solution_feasible(temp, capacity)):
                    temp.append(total_route_cost(temp,dist_matrix))
                    neighbours.append(temp)
        for i in combo[1][:-1]:
            for j in combo[0][:-1]:
                if i.id == 1 or j.id == 1:
                    continue
                temp = [list(item) for item in sol]
                c0 = list(combo[1])
                c1 = list(combo[0])
                index1 = temp.index(c0)
                index2 = temp.index(c1)
                c1.insert(c1.index(j),i)
                c0.insert(c0.index(i),j)
                c1.remove(j)
                c0.remove(i)
                temp[index1] = c0
                temp[index2] = c1
                if (is_solution_feasible(temp, capacity)):
                    temp.append(total_route_cost(temp,dist_matrix))
                    neighbours.append(temp)
    sorted_neighbours = sorted(neighbours, key=lambda x: x[-1])
    return sorted_neighbours

# Crea una lista de vecinos ordenados
# La creacion de vecinos se hace a travez del movimiento de nodos de una ruta a otra
# 
def create_neighbourhood2(sol, dist_matrix, capacity):
    neighbours = []
    for combo in list(combinations(sol, 2)):
        for i in combo[0][:-1]:
            for j in combo[1][:-1]:
                if j.id == 1:
                    continue
                temp = [list(item) for item in sol]
                c0 = list(combo[0])
                c1 = list(combo[1])
                index1 = temp.index(c0)
                index2 = temp.index(c1)
                c1.remove(j)
                c0.insert(c0.index(i)+1, j)
                temp[index1] = c0
                temp[index2] = c1
                if (is_solution_feasible(temp, capacity)):
                    temp.append(total_route_cost(temp,dist_matrix))
                    neighbours.append(temp)
        for i in combo[1][:-1]:
            for j in combo[0][:-1]:
                if j.id == 1:
                    continue
                temp = [list(item) for item in sol]
                c0 = list(combo[1])
                c1 = list(combo[0])
                index1 = temp.index(c0)
                index2 = temp.index(c1)
                c1.remove(j)
                c0.insert(c0.index(i)+1, j)
                temp[index1] = c0
                temp[index2] = c1
                if (is_solution_feasible(temp, capacity)):
                    temp.append(total_route_cost(temp,dist_matrix))
                    neighbours.append(temp)
    sorted_neighbours = sorted(neighbours, key=lambda x: x[-1])
    return sorted_neighbours

def tabu_search(solution, dist_matrix, iterations, capacity):
    best_solution_ever = solution
    best_cost_ever = total_route_cost(solution, dist_matrix)
    no_change = 0
    best_sol = solution
    best_cost = ()
    tabu_list = []
    for i in  range(iterations - 1):
        # Se sale si no hay cambios en la mejor solucion de todo el tiempo
        if no_change > 50:
            print("Exit: No change")
            break
        neighbourhood1 = create_neighbourhood1(best_sol, dist_matrix, capacity)
        neighbourhood2 = create_neighbourhood2(best_sol, dist_matrix, capacity)
        # Escogiendo el mejor vecino que no este en lista tabu
        for neighbour in neighbourhood1:
            if (neighbour not in tabu_list) or (neighbour[-1] < best_cost_ever):
                sol1 = neighbour
                break
            else:
                sol1 = -1
        for neighbour in neighbourhood2:
            if (neighbour not in tabu_list) or (neighbour[-1] < best_cost_ever):
                sol2 = neighbour
                break
            else:
                sol2 = -1
        if sol1 == -1 and sol2 == -1:
            print("Exit: No neighbours")
            break
        # Eligiendo la mejor solucion
        if sol1[-1] < sol2[-1]:
            best_sol = sol1[:-1]
            best_cost = sol1[-1]
            tabu_list.append(sol1)
        else:
            best_sol = sol2[:-1]
            best_cost = sol2[-1]
            tabu_list.append(sol2)
        # Tamaño maximo de lista tabu
        if len(tabu_list) > 5:
            tabu_list.pop(0)
        # Eligiendo la mejor solucion de todo el tiempo
        if best_cost < best_cost_ever:
            best_cost_ever = best_cost
            best_solution_ever = best_sol
            no_change = 0
            print("New best solution ever {0}, {1}".format(best_solution_ever, best_cost_ever))
        else:
            no_change += 1
        print("{0}.Best solution so far {1}, {2}".format(i, best_sol, best_cost))
    return best_solution_ever, best_cost_ever

if __name__ == "__main__":
    file_path = 'Instances\\P-n16-k8.vrp'
    capacity, clients = parse_vrp_file(file_path)
    sumDemand = sum(client.demand for client in clients)
    numTrucks = sumDemand//capacity
    trucks = []
    for i in range(numTrucks+3):
        trucks.append(Trucks(id=i+1, capacity=capacity))
    dist_matrix = calculate_distance_matrix(clients)
    create_initial_solution(trucks, clients, dist_matrix)
    initial_solution = []
    for truck in trucks:
        initial_solution.append(truck.route)
    initial_cost = total_route_cost(initial_solution, dist_matrix)
    opt_solution, opt_cost = tabu_search(initial_solution, dist_matrix, 500, capacity)
    print(opt_solution,opt_cost)
    print(is_solution_feasible(opt_solution,capacity))

