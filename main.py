from itertools import combinations
import numpy as np
import copy

class Trucks:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity # Capacidad del camion
        self.packages = 0 # Paquetes que lleva
        self.route = [] # Ruta con clientes
        self.current_time = 0 # Tiempo actual

    def __repr__(self):
        return (f"Truck(id={self.id}, capacity={self.capacity}, packages={self.packages}, "
                f"route={self.route}, current_time={self.current_time})")

class Clients:
    def __init__(self, id, x, y, demand, start_time, end_time, service):
        self.id = id
        self.x = x # Posicion del cliente
        self.y = y
        self.start_time = start_time # Ventanas de tiempo
        self.end_time = end_time
        self.demand = demand # Demanda del cliente
        self.service = service # Tiempo que demora en atender

    def __repr__(self):
        return f"{self.id}"

def calculate_distance_matrix(clients):
    num_clients = len(clients)
    distance_matrix = np.zeros((num_clients, num_clients)) # Matriz vacia

    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                distance_matrix[i][j] = np.sqrt((clients[i].x - clients[j].x) ** 2 + (clients[i].y - clients[j].y) ** 2) # Distancia euclidiana

    return distance_matrix

def read_data(file_name):
    trucks = []
    clients = []

    with open(file_name, 'r') as file:
        lines = file.readlines()

        vehicle_index = lines.index('VEHICLE\n') + 2
        vehicle_section = lines[vehicle_index].strip().split()
        num_trucks = int(vehicle_section[0])
        truck_capacity = int(vehicle_section[1])

        trucks = [Trucks(id=i, capacity=truck_capacity) for i in range(num_trucks)]

        customer_index = lines.index('CUSTOMER\n') + 2
        customer_section = lines[customer_index:]

        for line in customer_section:
            if line.strip():  # skip empty lines
                values = line.split()
                client = Clients(
                    id=int(values[0]),
                    x=float(values[1]),
                    y=float(values[2]),
                    demand=float(values[3]),
                    start_time=float(values[4]),
                    end_time=float(values[5]),
                    service=float(values[6])
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
        last_client = clients[0]  # Depot
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
            best_cost = float('inf')
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
    total_dist = 0
    traveled_time = 0
    for i in range(len(route) - 1):
        traveled_time = dist_matrix[route[i].id][route[i + 1].id]
        total_dist += max(traveled_time, route[i + 1].start_time) + route[i + 1].service
    return total_dist

def total_route_cost(solution, dist_matrix):
    return sum(route_cost(route, dist_matrix) for route in solution)

def is_route_feasible(route,capacity, dist_matrix):
    total_demand = sum(client.demand for client in route) # Demanda total de la ruta
    if total_demand > capacity:
        print("demand exit")
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



if __name__ == "__main__":
    trucks, clients = read_data("toy.txt")
    dist_matrix = calculate_distance_matrix(clients)

    create_initial_solution(trucks, clients, dist_matrix)
    solution = [truck.route for truck in trucks]

    print(total_route_cost(solution, dist_matrix))
    print("=======================")

    for route in solution:
        print(is_route_feasible(route, trucks[0].capacity, dist_matrix), route)



