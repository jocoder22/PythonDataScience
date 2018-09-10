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
plt.xlabel('My X label')
plt.ylabel('My Y label')
plt.title('My plot title')

plt.show()

# More on labels and title
label1 = 'First Label'
label2 = 'Second Label'
title = 'Main Title'


plt.scatter(x, y)
plt.xscale('log')

plt.xlabel(label1)
plt.ylable(label2)
plt.title(title)

plt.show()


# Changing tick value and labels
tvalues = []
tlabels = []


plt.xticks(tvalues, tlabels)
plt.show()
