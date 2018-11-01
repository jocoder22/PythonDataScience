import matplotlib.pyplot as plt
import numpy as np

x = [100, 400, 800, 874, 902, 482, 201, 305, 503, 591, 683, 496]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
z = [110, 826, 419, 223, 715, 932, 329, 214, 419, 835, 322, 148]
fac = ['Placebo', 'IP30mg', 'Placebo', 'IP30mg', 'Placebo',
       'IP10mg', 'Placebo', 'IP10mg', 'Placebo', 'IP20mg', 'Placebo', 'IP20mg']

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
plt.ylabel(label2)
plt.title(title)

plt.show()


# Changing tick value and labels
tvaluesx = []
tlabelsx = []

plt.xticks(tvaluesx, tlabelsx)
plt.yticks(y, ['Baseline', 'Visit1', 'Visit2', 'Visit3', 'Visit4',  'Visit5',
               'Visit6', 'Visit7', 'Visit8', 'Visit9', 'Visit10', 'Closeout'])
plt.show()

# More customization
# s= sets the size for the dots
zSizing = np.array(z)
plt.scatter(x, y, s=zSizing)

plt.show()


# Changing the color and opacity
# c= sets the color while alpha= sets the opacity(from 0.0 to 1.0)
# using dictionary to define the colors
mycolors = {
    'Placebo': 'red',
    'IP10mg': 'blue',
    'IP20mg': 'green',
    'IP30mg': 'yellow'
}


plt.scatter(x, y, s=zSizing, c=mycolors, alpha=6.2)
plt.text(406, 13, "Failed")
plt.text(508, 19, "lost")
plt.grid(True)
plt.show()
