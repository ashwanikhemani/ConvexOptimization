with open("result_lamda_1e-2.txt") as f:
	le2 = f.read().split("\n")[17:-6]

with open("result_lamda_1e-4.txt") as f:
	le4 = f.read().split("\n")[17:-6]

with open("result_lamda_1e-6.txt") as f:
	le6 = f.read().split("\n")[17:-6]

le2_x, le2_y, le4_x, le4_y, le6_x, le6_y = [], [], [], [], [], []

for it in le2:
	line = it.split(" ")
	le2_x.append(float(line[3]))
	le2_y.append(float(line[1]))

for it in le4:
	line = it.split(" ")
	le4_x.append(float(line[3]))
	le4_y.append(float(line[1]))

for it in le6:
	line = it.split(" ")
	le6_x.append(float(line[3]))
	le6_y.append(float(line[1]))

import matplotlib.pyplot as plt

#added these for asthetic reasons
le2_x.append(191.7557)
le2_y.append(6.38803453)
le4_x.append(191.7557)
le4_y.append(2.12936686)

plt.plot(le2_x, le2_y, "r")
plt.plot(le4_x, le4_y, "b")
plt.plot(le6_x, le6_y, "g")

plt.legend(["1e-2", "1e-4", "1e-6"])
plt.xlabel("CPU Time (seconds)")
plt.ylabel("Objective Value")

plt.show()
