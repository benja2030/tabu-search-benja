param numCustomers;
set Nodes := {1..numCustomers};
set Customers := {2..numCustomers};

param vehicles;
param vehCapacity;
param coords{Nodes, 1..2};
param demand{Customers};

param dist{i in Nodes, j in Nodes} := 
    sqrt((coords[i,1] - coords[j,1])^2 + (coords[i,2] - coords[j,2])^2);

var x{i in Nodes, j in Nodes, k in 1..vehicles} binary;
var u{Nodes} >= 0;

minimize z: sum{k in 1..vehicles, i in Nodes, j in Nodes} dist[i,j]*x[i,j,k];

s.t. visit_once {j in Customers}:
    sum {i in Nodes, k in 1..vehicles: i <> j} x[i,j,k] = 1;

s.t. start_depot {k in 1..vehicles}:
    sum {j in Customers} x[1,j,k] = 1;

s.t. flow_conservation {k in 1..vehicles, h in Customers}:
    sum {i in Nodes: i <> h} x[i,h,k] = sum {j in Nodes: j <> h} x[h,j,k];

s.t. end_depot {k in 1..vehicles}:
    sum {i in Customers} x[i,1,k] = 1;

s.t. capacity {k in 1..vehicles}:
    sum {i in Nodes, j in Customers: i <> j} demand[j] * x[i,j,k] <= vehCapacity;
    
s.t. subtour_elimination {i in Customers, j in Customers, k in 1..vehicles}:
	u[i] - u[j] + 1 <= (numCustomers - 1)*(1 - x[i,j,k])