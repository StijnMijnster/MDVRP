# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:02:02 2021

@author: stijn
"""

import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
import pandas as pd
import math

#%% ----- Problem -----

model = Model('Multiple Depot Vehicle Routing Problem')

#%% ----- Data -----

#time matrix input
data_input1 = pd.read_excel (r'C:/Users/stijn/Dropbox/Stijn/Research Project/Python/03 Final models/MDVRP/MDVRP_input.xlsx', sheet_name='Cost matrix')
data_input1 = data_input1.iloc[: , 1:]        #delete first column of pandas dataframe
tm = data_input1.values.tolist()              #time matrix

#depots input
data_input2 = pd.read_excel (r'C:/Users/stijn/Dropbox/Stijn/Research Project/Python/03 Final models/MDVRP/MDVRP_input.xlsx', sheet_name='Depots')
DEPOT_ID = data_input2['DEPOT_ID'].tolist()
DEPOT_xc = data_input2['XCOORD'].tolist()
DEPOT_yc = data_input2['YCOORD'].tolist()
DEPOT_S_TIME = data_input2['SERVICE_TIME'].tolist()
DEPOT_DEMAND = data_input2['DEMAND'].tolist()

#location input
data_input3 = pd.read_excel (r'C:/Users/stijn/Dropbox/Stijn/Research Project/Python/03 Final models/MDVRP/MDVRP_input.xlsx', sheet_name='Customers')
CUSTOMER_ID = data_input3['CUSTOMER_ID'].tolist()
CUSTOMER_xc = data_input3['XCOORD'].tolist()
CUSTOMER_yc = data_input3['YCOORD'].tolist()
CUSTOMER_S_TIME = data_input3['SERVICE_TIME'].tolist()
CUSTOMER_DEMAND = data_input3['DEMAND'].tolist()

LOC_ID = DEPOT_ID + CUSTOMER_ID
xc = DEPOT_xc + CUSTOMER_xc
yc = DEPOT_yc + CUSTOMER_yc
S_TIME = DEPOT_S_TIME + CUSTOMER_S_TIME
DEMAND = DEPOT_DEMAND + CUSTOMER_DEMAND

d = len(DEPOT_ID)                                           #number of depots
n = len(CUSTOMER_ID)                                        #number of customers
Q = 3                                                       #vehicle capacity
MNOVA = 5                                                  #maximum number of vehicles available 
M = 1000000

#%% ----- Sets and indices -----

D = [i for i in range (0,d)]                                #set of depots
N = [i for i in range (d,n+d)]                              #set of customers
V = D + N                                                   #set of nodes
A = [(i, j) for i in V for j in V if i !=j]                 #set of arcs
K = [i for i in range(MNOVA)]                               #set of vehicles

#%% ----- Parameters -----

c = {(i, j): np.hypot(xc[i]-xc[j],yc[i]-yc[j]) for i in V for j in V if i != j}         #euclidean distance
# c = {(i, j): tm[i][j] for i, j in A}                                                    #time matrix distance
s = S_TIME                                                                              #service time
q = {i: DEMAND[i] for i in V}                                                           #demand of the customer

#%% ----- Decision variables -----

#x(i,j,k) - The route of the truck from customer i to j, 1 = arc, 0 = no arc
x = {}
for i in V:
    for j in V:
        for k in K:
            x[i,j,k] = model.addVar (lb = 0, vtype = GRB.BINARY)

#T(i) - Time counter of elapsed time at the arrival of location i        
T = {}
for i in V:
    T[i] = model.addVar (lb = 0, vtype = GRB.CONTINUOUS)

#L(i) - Load counter after delivery at location i
L = {}
for i in V:
    L[i] = model.addVar (lb = 0, vtype = GRB.CONTINUOUS)

#%% ----- Objective function (minimize total distance) -----

Total_distance_travelled = quicksum (c[i,j]*x[i,j,k] for i in V for j in V if i != j for k in K)

model.setObjective (Total_distance_travelled)
model.modelSense = GRB.MINIMIZE
model.update ()

#%% ----- Constraints -----

#constraint 1: The sum of every row in the route variable x is equal to 1
con1 = {}
for i in N:
    con1[i,j,k] = model.addConstr(quicksum(x[i,j,k] for j in V for k in K) == 1)

#constraint 2: The sum of every column in the route variable x is equal to 1        
con2 = {}
for j in N:
    con2[i,j,k] = model.addConstr(quicksum(x[i,j,k] for i in V for k in K) == 1)

#constraint 3: Making sure the truck does not stay at the same location        
con3 = {}
for i in V:
    for k in K:
        con3[i,i,k] = model.addConstr(x[i,i,k] == 0)

#constraint 4: The sum of outgoing arcs is equal to the sum of incoming arcs
con4 = {} 
for j in V:
    for k in K:
        con4[i,j,k] = model.addConstr(quicksum(x[i,j,k] for i in V) == quicksum(x[j,i,k] for i in V))  

#constraint 5: Each vehicle depart from the depot only once
con5 = {}
for k in K:
    con5[k] = model.addConstr(quicksum (x[h,j,k] for j in N for h in D) <= 1)

#constraint 6: The timer at i plus the costs between i and j plus the service time at i is smaller or same as the timer at j
con6 = {}
for i in V:
    for j in V:
        if j >= d:
            if i != j:
                for k in K:
                    con6[i,j,k] = model.addConstr(T[i] + c[i,j]*x[i,j,k] + s[i] - M*(1-x[i,j,k]) <= T[j])
                
#constraint 7: The load after delivery at i minus the demand of j is greater or same as the load after delivery at j
con7 = {}
for i in V:
    for j in V:
        if j >= d:
            for k in K:
                con7[i,j,k] = model.addConstr(L[i] - q[j] + M*(1-x[i,j,k]) >= L[j])
            
#constraint 8: The load after delivery at i plus the demand of customer i is smaller or same as the capacity
con8 = {}
for i in V:
    con8[i] = model.addConstr(L[i] + q[i] <= Q)

#%% ----- Solve -----

model.update ()

model.setParam( 'OutputFlag', True)     # silencing gurobi output or not
model.setParam ('MIPGap', 0);           # find the optimal solution
model.write("output.lp")                # print the model in .lp format file

model.optimize ()

#%% ----- Results -----

print ('\n--------------------------------------------------------------------\n')
if model.status == GRB.Status.OPTIMAL:                          # If optimal solution is found
    print ('Minimal distance : %10.2f ' % model.objVal)
    print('\nFinished\n')
else:
    print ('\nNo feasible solution found\n')

active_arcs = [(i,j,k) for i in V for j in V if i != j for k in K if x[i,j,k].x == 1]
# print (('Route : '), sorted(active_arcs))

#sort active_arcs per route
active_arcs_i = [i for i,j,k in active_arcs]
active_arcs_j = [j for i,j,k in active_arcs]
active_arcs_k = [k for i,j,k in active_arcs]

active_arcs_kij = []
for a in range(len(active_arcs)):
    active_arcs_sorted = (active_arcs_k[a], active_arcs_i[a], active_arcs_j[a])
    active_arcs_kij.append(active_arcs_sorted)

print('Route (kij) : ')
active_arcs_kij_sorted = sorted(active_arcs_kij)
print("")

for b1 in range(len(active_arcs)):
    for b2 in range(len(K)):
        if active_arcs_kij_sorted[b1][0] == b2:
            print(active_arcs_kij_sorted[b1])

#make graph
fig, ax = plt.subplots(figsize=(6,6))
plt.xlim([0, 10])
plt.ylim([0, 10])

for i, j, k in active_arcs:
    plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='grey', zorder=0)
for d1 in range(d):
    plt.plot(xc[d1], yc[d1], c='black', marker='s', markersize=10)
plt.scatter(xc[d:], yc[d:], c='purple')

#plot LOC_ID next to points
for i, txt in enumerate(LOC_ID):
    plt.annotate(txt, (xc[i], yc[i]), xytext=(xc[i]+0.12, yc[i]+0.25), bbox=dict(boxstyle="round", alpha=0.1))

print()

#count number of vehicles necessary
total_vehicles = 0
active_arcs_x = [x for x,y,k in active_arcs]
for d1 in range(d):
    vehicles = active_arcs_x.count(d1)
    total_vehicles = total_vehicles + vehicles
    print("Number of vehicles departing from depot", d1, ": ", vehicles)
    
print("                 Total vehicles necessary : ", total_vehicles)

#print decision variable T = Time counter (TC)
print ('\nArrival time at location:')
for d1 in range(d):
    print ('%35.0f' % LOC_ID[d1] + '%8.0f' % T[d1].x)

for i in N: 
    AT_C1 = LOC_ID[i]               #arrival time column 1
    AT_C2 = T[i].x                  #arrival time column 2
    AT = '%35.0f' % AT_C1 + '%8.0f' % AT_C2
    print(AT)

#print decision variable L = Load counter (LC)
print ('\nLoad after delivery at location:')
for d1 in range(d):
    print ('%35.0f' % LOC_ID[d1] + '%8.0f' % L[d1].x)
    
for i in N:
    LC_C1 = LOC_ID[i]               #load counter column 1
    LC_C2 = L[i].x                  #load counter column 2
    LC = '%35.0f' % LC_C1 + '%8.0f' % LC_C2
    print(LC)









