with open("lbfgs-le-4.txt") as f:
	le4 = f.read().split("\n")[24:-6]

with open("adam-1-0.0001.txt") as f:
	a_le4 = f.read().split("\n")[1:]

with open("sgd-0.01-0.0001.txt") as f:
	s_le4 = f.read().split("\n")[1:-1]

le4_x, le4_y , a_le4_x, a_le4_y, s_le4_x, s_le4_y= [], [], [], [], [], []

#1e-4

for it in le4:
    line = it.split("\t")
    le4_x.append(float(line[4]))
    le4_y.append(float(line[8]))

for it in a_le4:
    line = it.split("\t")
    a_le4_x.append(int(line[0]))
    a_le4_y.append(float(line[2]))


for it in s_le4:
    line = it.split("\t")
    s_le4_x.append(int(line[0]))
    s_le4_y.append(float(line[3]))
import matplotlib.pyplot as plt


#1e-4

a_le4_y = [i * 100 for i in a_le4_y]
s_le4_y = [i * 100 for i in s_le4_y]

plt.plot(le4_x, le4_y, "r")
plt.plot(a_le4_x, a_le4_y, "b")
plt.plot(s_le4_x, s_le4_y, "g")

plt.legend(["lbfgs", "adam", "sgd"])
plt.xlabel("number of passes")
plt.ylabel("word error")

plt.show()

