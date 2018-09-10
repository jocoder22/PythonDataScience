import matplotlib.pyplot as plt

x = []
y = []
# line plot
plt.plot(x, y)
plt.show()


# Scatter plot
plt.scatter(x, y)
plt.xscale('log')
plt.show()


# Histogram
plt.hist(x, bin=10)
plt.show()
plt.clf()  # this clears the histogram


# Plot customization
plt.plot(x, y)
plt.xlable('My X label')
plt.ylable('My Y label')
plt.title('My plot title')

plt.show()

