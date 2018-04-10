with open("lbfgs-le-2.txt") as f:
	le2 = f.read().split("\n")[24:-6]

with open("adam-1e-2.txt") as f:
	a_le2 = f.read().split("\n")[2:-1]

with open("sgd-1e-2.txt") as f:
	s_le2 = f.read().split("\n")[1:-1]

with open("lbfgs-le-4.txt") as f:
	le4 = f.read().split("\n")[24:-6]

with open("adam-1e-4.txt") as f:
	a_le4 = f.read().split("\n")[2:-1]

with open("sgd-1e-4.txt") as f:
	s_le4 = f.read().split("\n")[3:-1]

with open("lbfgs-le-6.txt") as f:
	le6 = f.read().split("\n")[24:-6]

with open("adam-1e-6.txt") as f:
	a_le6 = f.read().split("\n")[2:-1]

with open("sgd-1e-6.txt") as f:
	s_le6 = f.read().split("\n")[3:-1]
    
le2_x, le2_y, le4_x, le4_y, le6_x, le6_y = [], [], [], [], [], []
a_le2_x, a_le2_y, a_le4_x, a_le4_y, a_le6_x, a_le6_y = [], [], [], [], [], []
s_le2_x, s_le2_y, s_le4_x, s_le4_y, s_le6_x, s_le6_y = [], [], [], [], [], []

#1e-2

for it in le2:
    line = it.split("\t")
    le2_x.append(float(line[4]))
    le2_y.append(float(line[1]))

for it in a_le2:
    line = it.split(":")
    a_le2_x.append(int(line[0]))
    a_le2_y.append(float(line[1]))

for it in s_le2:
    line = it.split(":")
    s_le2_x.append(int(line[0]))
    s_le2_y.append(float(line[1]))

#1e-4

for it in le4:
    line = it.split("\t")
    le4_x.append(float(line[4]))
    le4_y.append(float(line[1]))

for it in a_le4:
    line = it.split(":")
    a_le4_x.append(int(line[0]))
    a_le4_y.append(float(line[1]))

for it in s_le4:
    line = it.split(":")
    s_le4_x.append(int(line[0]))
    s_le4_y.append(float(line[1]))

#1e-6

for it in le6:
    line = it.split("\t")
    le6_x.append(float(line[4]))
    le6_y.append(float(line[1]))

for it in a_le6:
    line = it.split(":")
    a_le6_x.append(int(line[0]))
    a_le6_y.append(float(line[1]))

for it in s_le6:
    line = it.split(":")
    s_le6_x.append(int(line[0]))
    s_le6_y.append(float(line[1]))


import matplotlib.pyplot as plt

#1e-2

plt.plot(le2_x, le2_y, "r")
plt.plot(a_le2_x, a_le2_y, "b")
plt.plot(s_le2_x, s_le2_y, "g")

plt.legend(["lbfgs", "adam", "sgd"])
plt.xlabel("number of passes")
plt.ylabel("Objective Value")

plt.show()

#1e-4

plt.figure()
plt.plot(le4_x, le4_y, "r")
plt.plot(a_le4_x, a_le4_y, "b")
plt.plot(s_le4_x, s_le4_y, "g")

plt.legend(["lbfgs", "adam", "sgd"])
plt.xlabel("number of passes")
plt.ylabel("Objective Value")

plt.show()

#1e-6
plt.figure()

plt.plot(le6_x, le6_y, "r")
plt.plot(a_le6_x, a_le6_y, "b")
plt.plot(s_le6_x, s_le6_y, "g")

plt.legend(["lbfgs", "adam", "sgd"])
plt.xlabel("number of passes")
plt.ylabel("Objective Value")

plt.show()

