from itertools import combinations
import numpy as np


class Trucks:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity  # Capacidad del camion
        self.packages = 0  # Paquetes que lleva
        self.route = []  # Ruta con clientes
        self.current_time = 0  # Tiempo actual

    def __repr__(self):
        return (
            f"Truck(id={self.id}, capacity={self.capacity}, packages={self.packages}, "
            f"route={self.route}, current_time={self.current_time})"
        )


class Clients:
    def __init__(self, id, x, y, demand, start_time, end_time, service):
        self.id = id
        self.x = x  # Posicion del cliente
        self.y = y
        self.start_time = start_time  # Ventanas de tiempo
        self.end_time = end_time
        self.demand = demand  # Demanda del cliente
        self.service = service  # Tiempo que demora en atender

    def __repr__(self):
        return f"{self.id}"


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


def read_data(file_name):
    trucks = []
    clients = []

    with open(file_name, "r") as file:
        lines = file.readlines()

        vehicle_index = lines.index("VEHICLE\n") + 2
        vehicle_section = lines[vehicle_index].strip().split()
        num_trucks = int(vehicle_section[0])
        truck_capacity = int(vehicle_section[1])

        trucks = [Trucks(id=i, capacity=truck_capacity) for i in range(num_trucks)]

        customer_index = lines.index("CUSTOMER\n") + 2
        customer_section = lines[customer_index:]

        for line in customer_section:
            if line.strip():
                values = line.split()
                client = Clients(
                    id=int(values[0]),
                    x=float(values[1]),
                    y=float(values[2]),
                    demand=float(values[3]),
                    start_time=float(values[4]),
                    end_time=float(values[5]),
                    service=float(values[6]),
                )
                clients.append(client)

    return trucks, clients


def is_feasible(truck, client, distance_matrix):
    # Revisa si el camion puede con la demanda del cliente
    if truck.packages + client.demand > truck.capacity:
        return False

    # Define el ultimo cliente visitado
    if not truck.route:
        last_client = clients[0]
    else:
        last_client = truck.route[-1]
    # Calcula tiempos de viaje
    travel_time = distance_matrix[last_client.id][client.id]
    arrival_time = truck.current_time + travel_time

    # Revisa restricciones de ventanas de tiempo
    if arrival_time > client.end_time:
        return False
    return True


def add_client_to_truck(truck, client, distance_matrix):
    # Actualiza la carga del camion y su ruta
    truck.packages += client.demand
    truck.route.append(client)

    # Calcula el tiempo de viaje hasta el cliente
    if len(truck.route) == 1:
        last_client = clients[0]  # Almacen
    else:
        last_client = truck.route[-2]

    travel_time = distance_matrix[last_client.id][client.id]
    arrival_time = truck.current_time + travel_time

    # Actualiza el tiempo del camion
    truck.current_time = max(arrival_time, client.start_time) + client.service


def check_costs(truck, client, dist_matrix):
    if len(truck.route) == 1:
        last_client = clients[0]
    else:
        last_client = truck.route[-1]
    print(last_client.id,client.id)
    travel_time = dist_matrix[last_client.id][client.id]
    # Retorna el coste que tiene el viaje hasta un cliente
    return travel_time + client.service


def create_initial_solution(trucks, clients, dist_matrix):
    # Agrega los almacenes a las rutas
    for truck in trucks:
        add_client_to_truck(truck, clients[0], dist_matrix)
    # Clientes no visitados
    unvisited_clients = clients[1:]
    # Variable para revisar si se realizo algun cambio
    change = False

    # Se crean rutas para cada camion
    for truck in trucks:
        # Se agregan clientes mientras que queden clientes sin visitar
        while unvisited_clients:
            best_cost = float("inf")
            best_client = None
            for client in unvisited_clients:
                if is_feasible(truck, client, dist_matrix):
                    current_cost = check_costs(truck, client, dist_matrix)
                    if current_cost < best_cost:
                        change = True
                        best_cost = current_cost
                        best_client = client
            if not change:
                print(f"Can't add more clients {truck}")
                break
            add_client_to_truck(truck, best_client, dist_matrix)
            unvisited_clients.remove(best_client)
            change = False
        add_client_to_truck(truck, clients[0], dist_matrix)


def route_cost(route, dist_matrix):
    traveled_distance = 0
    #total_time = 0
    for i in range(len(route) - 1):
        traveled_distance += dist_matrix[route[i].id][route[i + 1].id]
        #total_time += max(traveled_distance, route[i + 1].start_time) + route[i + 1].service
    return traveled_distance

def total_route_cost(solution, dist_matrix):
    return sum(route_cost(route, dist_matrix) for route in solution)

def is_route_feasible(route, capacity, dist_matrix):
    total_demand = sum(client.demand for client in route)  # Demanda total de la ruta
    if total_demand > capacity:
        return False

    current_time = 0
    last_client = route[0]
    for client in route:
        travel_time = dist_matrix[last_client.id][client.id]
        arrival_time = current_time + travel_time

        if arrival_time > client.end_time:
            return False

        current_time = max(arrival_time, client.start_time) + client.service
        last_client = client

    return True

def is_solution_feasible(solution, capacity, dist_matrix):
    for route in solution:
        route_demand = sum(client.demand for client in route)
        if route_demand > capacity:
            return False
        current_time = 0
        last_client = route[0]
        counter = 0
        for client in route:
            travel_time = dist_matrix[last_client.id][client.id]
            arrival_time = current_time + travel_time
            if arrival_time > client.end_time:
                return False
            current_time = max(arrival_time, client.start_time) + client.service
            last_client = client
            counter += 1
    return True


def create_neighbourhood1(sol, dist_matrix, capacity):
    neighbours = []
    for combo in list(combinations(sol, 2)):
        for i in combo[0][:-1]:
            for j in combo[1][:-1]:
                if i.id == 0 or j.id == 0:
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
                if (is_solution_feasible(temp, capacity, dist_matrix)):
                    temp.append(total_route_cost(temp,dist_matrix))
                    neighbours.append(temp)
        for i in combo[1][:-1]:
            for j in combo[0][:-1]:
                if i.id == 0 or j.id == 0:
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
                if (is_solution_feasible(temp, capacity, dist_matrix)):
                    temp.append(total_route_cost(temp,dist_matrix))
                    neighbours.append(temp)
    sorted_neighbours = sorted(neighbours, key=lambda x: x[-1])
    return sorted_neighbours

def create_neighbourhood2(sol, dist_matrix, capacity):
    neighbours = []
    for combo in list(combinations(sol, 2)):
        for i in combo[0][:-1]:
            for j in combo[1][:-1]:
                if j.id == 0:
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
                if (is_solution_feasible(temp, capacity, dist_matrix)):
                    temp.append(total_route_cost(temp,dist_matrix))
                    neighbours.append(temp)
        for i in combo[1][:-1]:
            for j in combo[0][:-1]:
                if j.id == 0:
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
                if (is_solution_feasible(temp, capacity, dist_matrix)):
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
        if no_change > 15:
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
        if len(tabu_list) > 10:
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
    trucks, clients = read_data("C101.txt")
    dist_matrix = calculate_distance_matrix(clients)
    capacity = trucks[0].capacity
    create_initial_solution(trucks, clients, dist_matrix)
    initial_solution = [truck.route for truck in trucks]

    opt_solution, opt_cost = tabu_search(initial_solution, dist_matrix, 100, capacity)
    print(opt_solution, opt_cost)

