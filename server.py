import numpy as np
import matplotlib.pyplot as plt

print("Please enter the server names and sizes when prompted as such -> X:4, Y:2, Z:7")
input1 = input("SERVERS:")

servString = input1.split(',')
names = []
sizes = []
total = 0

for i in range(len(servString)):
    servString[i] = servString[i].strip()
    names.append(servString[i].split(':')[0])
    sizes.append(int(servString[i].split(':')[1]))

servers = dict(zip(names, sizes))

for item in servers:
    total += servers[item]

print(total)

