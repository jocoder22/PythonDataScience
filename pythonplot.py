import matplotlib.pyplot as plt

# line plot
plt.plot(x, y)
plt.show()


# Scatter plot
plt.scatter(x, y)
plt.xscale('log')
plt.show()


# Histogram
plt.hist(x, bin = 10)
plt.show()
