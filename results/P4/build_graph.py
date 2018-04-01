threshold = 1.001*2.1278768

with open("result_1core.txt") as f:
	core1 = f.read().split("\n")[17:-6]

with open("result_2core.txt") as f:
	core2 = f.read().split("\n")[17:-6]

with open("result_4core.txt") as f:
	core4 = f.read().split("\n")[17:-6]

with open("result_6core.txt") as f:
	core6 = f.read().split("\n")[17:-6]

with open("result_8core.txt") as f:
	core8 = f.read().split("\n")[17:-6]

c1_t, c2_t, c4_t, c6_t, c8_t = 0.0, 0.0, 0.0, 0.0, 0.0

for it in core1:
	line = it.split(" ")
	if float(line[1]) < threshold:
		c1_t = float(line[3])
		break

for it in core2:
	line = it.split(" ")
	if float(line[1]) < threshold:
		c2_t = float(line[3])
		break

for it in core4:
	line = it.split(" ")
	if float(line[1]) < threshold:
		c4_t = float(line[3])
		break

for it in core6:
	line = it.split(" ")
	if float(line[1]) < threshold:
		c6_t = float(line[3])
		break

for it in core8:
	line = it.split(" ")
	if float(line[1]) < threshold:
		c8_t = float(line[3])
		break

import matplotlib.pyplot as plt

plt.plot([1, 2, 4, 6, 8], [c1_t/c1_t, c1_t/c2_t, c1_t/c4_t, c1_t/c6_t, c1_t/c8_t], "b")
plt.xlabel("Number of cores")
plt.ylabel("Speedup")
plt.show()
