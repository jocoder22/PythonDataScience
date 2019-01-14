import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

irisurl = "http://aima.cs.berkeley.edu/data/iris.csv"
iris_data = pd.read_csv(irisurl, sep=',', decimal='.', header=None,
                        names=['sepal_length', 'sepal_width', 'petal_length',
                               'petal_width', 'target'])

vcl = iris_data.target == 'versicolor'
versicolor_pl = iris_data.petal_length[vcl]
versicolor_pw = iris_data.petal_width[vcl]

# Set default Seaborn style
sns.set()


def ecdf(data):
    """ Compute ECDF for one-dimensional array"""
    n = len(data)

    x = np.sort(data)
    y = np.arange(1, n + 1) / n

    return x, y


np.random.seed(42)

# Compute mean  and standard deviation
mu = np.mean(versicolor_pl)
sigma = np.std(versicolor_pl)

# Draw out of an normal distribution with parameter mu and sigma
versicolor_theo = np.random.normal(mu, sigma, 100000)

bins = int(np.sqrt(len(versicolor_theo)))

# Plot the PDF and label axes
_ = plt.hist(versicolor_theo,
             normed=True, bins=bins, histtype='step')
_ = plt.xlabel('Versicolor petal length')
_ = plt.ylabel('PDF')

# Show the plot
plt.pause(3)
plt.clf()


for _ in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(versicolor_pl, size=len(versicolor_pl))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(versicolor_pl)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('Versicolor petal length (cm)')
_ = plt.ylabel('ECDF')

plt.pause(2)
plt.clf()



# Plot the illiteracy rate versus fertility
_ = plt.plot(versicolor_pl, versicolor_pw, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('versicolor Petal Length')
_ = plt.ylabel('versicolor Petal Width')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(versicolor_pl, versicolor_pw, 1)

# Print the results to the screen
print('slope =', a, 'versicolor Petal Width / percent versicolor Petal Length')
print('intercept =', b, 'versicolor Petal Width')

# Make theoretical line to plot
x = np.array([np.min(versicolor_pl) -0.1, np.max(versicolor_pl) + 0.1])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()
# Show the plot
# plt.pause(2)
# plt.close()
# plt.show()
