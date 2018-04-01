with open("best_lambda.txt") as f:
	bl = f.read().split("\n")[17:-6]
	
time, train_l, train_w, test_l, test_w = [], [], [], [], []

for it in bl:
	line = it.split(" ")
	time.append(float(line[3]))
	train_l.append(float(line[-4]))
	train_w.append(float(line[-3]))
	test_l.append(float(line[-2]))
	test_w.append(float(line[-1]))

import matplotlib.pyplot as plt

plt.subplot(2, 1, 1)
plt.plot(time, train_l, "b-", time, test_l, "g-")
plt.legend(["Train", "Test"])
plt.ylabel("Letter-Wise Error")

plt.subplot(2, 1, 2)
plt.plot(time, train_w, "b-", time, test_w, "g-")
plt.legend(["Train", "Test"])
plt.xlabel("CPU Time (seconds)")
plt.ylabel("Word-Wise Error")

plt.show()
