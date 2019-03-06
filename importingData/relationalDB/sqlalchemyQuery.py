# #####################creating database
from sklearn import preprocessing
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sqlalchemy import Table, Column, String, Integer, Float, Boolean

# Define a new table with a name, count, amount, and valid column: data
data = Table('data', metadata,
             Column('name', String(255)),
             Column('count', Integer()),
             Column('amount', Float()),
             Column('valid', Boolean())
)

# Use the metadata to create the table
metadata.create_all(engine)

# Print table details
print(repr(data))



# Define a new table with a name, count, amount, and valid column: data
data = Table('data', metadata,
             Column('name', String(255), unique=True),
             Column('count', Integer(), default=1),
             Column('amount', Float()),
             Column('valid', Boolean(), default=False)
)

# Use the metadata to create the table
metadata.create_all(engine)

# Print the table details
print(repr(metadata.tables['data']))



############### inserting dataset
# Import insert and select from sqlalchemy
from sqlalchemy import select, insert

# Build an insert statement to insert a record into the data table: stmt
stmt=insert(data).values(name='Anna', count=1,
                            amount=1000.00, valid=True)

# Execute the statement via the connection: results
results=connection.execute(stmt)

# Print result rowcount
print(results.rowcount)

# Build a select statement to validate the insert
stmt=select([data]).where(data.columns.name == 'Anna')

# Print the result of executing the query.
print(connection.execute(stmt).first())



# Build a list of dictionaries: values_list
values_list = [
    {'name': 'Anna', 'count': 1, 'amount': 1000.00, 'valid': True},
    {'name': 'Taylor', 'count': 1, 'amount': 750.00, 'valid': False}
]

# Build an insert statement for the data table: stmt
stmt = insert(data)

# Execute stmt with the values_list: results
results = connection.execute(stmt, values_list)

# Print rowcount
print(results.rowcount)



# Create an empty list and zeroed row count: values_list, total_rowcount
values_list = []
total_rowcount = 0

# Enumerate the rows of csv_reader
for idx, row in enumerate(csv_reader):
    #create data and append to values_list
    data = {'state': row[0], 'sex': row[1], 'age': row[2], 'pop2000': row[3],
            'pop2008': row[4]}
    values_list.append(data)

    # Check to see if divisible by 51
    if idx % 51 == 0:
        results = connection.execute(stmt, values_list)
        total_rowcount += results.rowcount
        values_list = []

# Print total rowcount
print(total_rowcount)



# Build a select statement: select_stmt
select_stmt = select([state_fact]).where(state_fact.columns.name == 'New York')

# Print the results of executing the select_stmt
print(connection.execute(select_stmt).fetchall())

# Build a statement to update the fips_state to 36: stmt
stmt = update(state_fact).values(fips_state = 36)

# Append a where clause to limit it to records for New York state
stmt = stmt.where(state_fact.columns.name == 'New York')

# Execute the statement: results
results = connection.execute(stmt)

# Print rowcount
print(results.rowcount)

# Execute the select_stmt again to view the changes
print(connection.execute(select_stmt).fetchall())

        # Build a statement to update the notes to 'The Wild West': stmt
        stmt=update(state_fact).values(notes='The Wild West')

        # Append a where clause to match the West census region records
        stmt=stmt.where(state_fact.columns.census_region_name == 'West')

        # Execute the statement: results
        results=connection.execute(stmt)

        # Print rowcount
        print(results.rowcount)


# Build a statement to select name from state_fact: stmt
fips_stmt = select([state_fact.columns.name])

# Append a where clause to Match the fips_state to flat_census fips_code
fips_stmt = fips_stmt.where(
    state_fact.columns.fips_state == flat_census.columns.fips_code)

# Build an update statement to set the name to fips_stmt: update_stmt
update_stmt = update(flat_census).values(state_name=fips_stmt)

# Execute update_stmt: results
results = connection.execute(update_stmt)

# Print rowcount
print(results.rowcount)



########################## Deletig a table
# Import delete, select
from sqlalchemy import select, delete

# Build a statement to empty the census table: stmt
stmt = delete(census)

# Execute the statement: results
results = connection.execute(stmt)

# Print affected rowcount
print(results.rowcount)

# Build a statement to select all records from the census table
stmt = select([census])

# Print the results of executing the statement to verify there are no rows
print(connection.execute(stmt).fetchall())


# Build a statement to count records using the sex column for Men ('M') age 36: stmt
stmt = select([func.count(census.columns.sex)]).where(
    and_(census.columns.sex == 'M',
         census.columns.age == 36)
)

# Execute the select statement and use the scalar() fetch method to save the record count
to_delete = connection.execute(stmt).scalar()

# Build a statement to delete records from the census table: stmt_del
stmt_del = delete(census)


# Append a where clause to target Men ('M') age 36
stmt_del = stmt_del.where(
    and_(census.columns.sex == 'M',
         census.columns.age == 36)
)


# Execute the statement: results
results = connection.execute(stmt_del)



# Drop the state_fact tables
state_fact.drop(engine)

# Check to see if state_fact exists
print(state_fact.exists(engine))

# Drop all tables
metadata.drop_all(engine)

# Check to see if census exists
print(census.exists(engine))





# final project
# Import create_engine, MetaData

from sqlalchemy import create_engine, MetaData 
# Define an engine to connect to chapter5.sqlite: engine
engine = create_engine('sqlite:///chapter5.sqlite')

# Initialize MetaData: metadata
metadata = MetaData()


from sqlalchemy import Table, Column, String, Integer

# Build a census table: census
census = Table('census', metadata,
               Column('state', String(30)),
               Column('sex', String(1)),
               Column('age', Integer()),
               Column('pop2000', Integer()),
               Column('pop2008', Integer()))

# Create the table in the database
metadata.create_all(engine)

values_list = []


import csv
csv_reader = csv.reader(csvfile)
# Iterate over the rows
for row in csv_reader:
    # Create a dictionary with the values
    data = {'state': row[0], 'sex': row[1], 'age':row[2], 'pop2000': row[3],
            'pop2008': row[4]}
    # Append the dictionary to the values list
    values_list.append(data)


from sqlalchemy import insert

# Build insert statement: stmt
stmt = insert(census)

# Use values_list to insert data: results
results = connection.execute(stmt, values_list)

# Print rowcount
print(results.rowcount)



from sqlalchemy import select

# Calculate weighted average age: stmt
stmt = select([census.columns.sex,
               (func.sum(census.columns.pop2008 * census.columns.age) /
                func.sum(census.columns.pop2008)).label('average_age')
               ])

# Group by sex
stmt = stmt.group_by(census.columns.sex)

# Execute the query and store the results: results
results = connection.execute(stmt).fetchall()

# Print the average age by sex
for result in results:
    print(result.sex, result.average_age)


# import case, cast and Float from sqlalchemy
from sqlalchemy import case, cast, Float

# Build a query to calculate the percentage of females in 2000: stmt
stmt = select([census.columns.state,
    (func.sum(
        case([
            (census.columns.sex == 'F', census.columns.pop2000)
        ], else_=0)) /
     cast(func.sum(census.columns.pop2000), Float) * 100).label('percent_female')
])

# Group By state
stmt = stmt.group_by(census.columns.state)

# Execute the query and store the results: results
results = connection.execute(stmt).fetchall()

# Print the percentage
for result in results:
    print(result.state, result.percent_female)


# Build query to return state name and population difference from 2008 to 2000
stmt = select([census.columns.state,
     (census.columns.pop2008 - census.columns.pop2000).label('pop_change')
])

# Group by State
stmt = stmt.group_by(census.columns.state)

# Order by Population Change
stmt = stmt.order_by(desc('pop_change'))

# Limit to top 10
stmt = stmt.limit(10)

# Use connection to execute the statement and fetch all results
results = connection.execute(stmt).fetchall()

# Print the state and population change for each record
for result in results:
    print('{}:{}'.format(result.state, result.pop_change))


#################### ploting
# Import matplotlib.pyplot

# Set the style to 'ggplot'
plt.style.use('ggplot')

# Create a figure with 2x2 subplot layout
plt.subplot(2, 2, 1)

# Plot the enrollment % of women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Plot the enrollment % of women in Computer Science
plt.subplot(2, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Add annotation
cs_max = computer_science.max()
yr_max = year[computer_science.argmax()]
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(
    yr_max-1, cs_max-10), arrowprops=dict(facecolor='black'))

# Plot the enrollmment % of women in Health professions
plt.subplot(2, 2, 3)
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Plot the enrollment % of women in Education
plt.subplot(2, 2, 4)
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve spacing between subplots and display them
plt.tight_layout()
plt.show()


plt.plot(year, computer_science, color='red', label='Computer Science')
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
plt.legend(loc='lower right')

# Compute the maximum enrollment of women in Computer Science: cs_max
cs_max = computer_science.max()

# Calculate the year in which there was maximum enrollment of women in Computer Science: yr_max
yr_max = year[computer_science.argmax()]

# Add a black arrow annotation
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(
    yr_max+5, cs_max+5), arrowprops=dict(facecolor='black'))

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()


plt.pcolor(A, cmap='Blues')
plt.colorbar()
plt.show()


# Generate a 2-D histogram
plt.hist2d(hp, mpg, bins=(20, 20), range=((40, 235), (8, 48)))

# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()






plt.hexbin(hp, mpg, gridsize=(15, 12), extent=(40, 235, 8, 48))

# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')
plt.show()


# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Print the shape of the image
print(img.shape)

# Compute the sum of the red, green and blue channels: intensity
intensity = img.sum(axis=2)

# Print the shape of the intensity
print(intensity.shape)

# Display the intensity with a colormap of 'gray'
plt.imshow(intensity, cmap='gray')

# Add a colorbar
plt.colorbar()

# Hide the axes and show the figure
plt.axis('off')
plt.show()





# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Specify the extent and aspect ratio of the top left subplot
plt.subplot(2, 2, 1)
plt.title('extent=(-1,1,-1,1),\naspect=0.5')
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.imshow(img, extent=(-1, 1, -1, 1), aspect=0.5)

# Specify the extent and aspect ratio of the top right subplot
plt.subplot(2, 2, 2)
plt.title('extent=(-1,1,-1,1),\naspect=1')
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.imshow(img, extent=(-1, 1, -1, 1), aspect=1)

# Specify the extent and aspect ratio of the bottom left subplot
plt.subplot(2, 2, 3)
plt.title('extent=(-1,1,-1,1),\naspect=2')
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.imshow(img, extent=(-1, 1, -1, 1), aspect=2)

# Specify the extent and aspect ratio of the bottom right subplot
plt.subplot(2, 2, 4)
plt.title('extent=(-2,2,-1,1),\naspect=2')
plt.xticks([-2, -1, 0, 1, 2])
plt.yticks([-1, 0, 1])
plt.imshow(img, extent=(-2, 2, -1, 1), aspect=2)

# Improve spacing and display the figure
plt.tight_layout()
plt.show()


# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Extract minimum and maximum values from the image: pmin, pmax
pmin, pmax = image.min(), image.max()
print("The smallest & largest pixel intensities are %d & %d." % (pmin, pmax))

# Rescale the pixels: rescaled_image
rescaled_image = 256*(image-pmin) / (pmax-pmin)
print("The rescaled smallest & largest pixel intensities are %.1f & %.1f." %
      (rescaled_image.min(), rescaled_image.max()))

# Display the original image in the top subplot
plt.subplot(2, 1, 1)
plt.title('original image')
plt.axis('off')
plt.imshow(image)

# Display the rescaled image in the bottom subplot
plt.subplot(2, 1, 2)
plt.title('rescaled image')
plt.axis('off')
plt.imshow(rescaled_image)

plt.show()


# Import plotting modules

# Plot a linear regression between 'weight' and 'hp'
sns.lmplot(x='weight', y='hp', data=auto)

# Display the plot
plt.show()


# Generate a green residual plot of the regression between 'hp' and 'mpg'
sns.residplot(x='hp', y='mpg', data=auto, color='green')

# Display the plot
plt.show()


# Generate a scatter plot of 'weight' and 'mpg' using red circles
plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')

# Plot in blue a linear regression of order 1 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto,
            scatter=None, color='blue', label='order 1')

# Plot in green a linear regression of order 2 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, scatter=None,
            order=2, color='green', label='order 2')

# Add a legend and display the plot
plt.legend(loc='upper right')
plt.show()


# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x='weight', y='mpg', data=auto, palette='Set1', hue='origin')

# Display the plot
plt.show()


# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x='weight', y='hp', data=auto, palette='Set1', hue='origin')

# Display the plot
plt.show()


# Plot linear regressions between 'weight' and 'hp' grouped row-wise by 'origin'
sns.lmplot(x='weight', y='hp', data=auto, palette='Set1', row='origin')

# Display the plot
plt.show()


# Make a strip plot of 'hp' grouped by 'cyl'
plt.subplot(2, 1, 1)
sns.stripplot(x='cyl', y='hp', data=auto)

# Make a strip plot of 'hp' grouped by 'cyl'
plt.subplot(2, 1, 1)
sns.stripplot(x='cyl', y='hp', data=auto)

# Make the strip plot again using jitter and a smaller point size
plt.subplot(2, 1, 2)
sns.stripplot(x='cyl', y='hp', data=auto, size=3, jitter=True)

# Display the plot
plt.show()


# Generate a swarm plot of 'hp' grouped horizontally by 'cyl'
plt.subplot(2, 1, 1)
sns.swarmplot(x='cyl', y='hp', data=auto)

# Generate a swarm plot of 'hp' grouped vertically by 'cyl' with a hue of 'origin'
plt.subplot(2, 1, 2)
sns.swarmplot(x='hp', y='cyl', data=auto, hue='origin', orient='h')

# Display the plot
plt.show()


# Generate a violin plot of 'hp' grouped horizontally by 'cyl'
plt.subplot(2, 1,1)
sns.violinplot(x='cyl', y='hp', data=auto)

# Generate the same violin plot again with a color of 'lightgray' and without inner annotations
plt.subplot(2, 1, 2)
sns.violinplot(x='cyl', y='hp', data=auto, color='lightgray', inner=None)

# Overlay a strip plot on the violin plot
sns.stripplot(x='cyl', y='hp', data=auto, size=1.5, jitter=True)

# Display the plot
plt.show()


# Generate a joint plot of 'hp' and 'mpg'
sns.jointplot(x='hp', y='mpg', data=auto)

# Display the plot
plt.show()


# Generate a joint plot of 'hp' and 'mpg' using a hexbin plot
sns.jointplot(x='hp', y='mpg', data=auto, kind='hex')

# kind = 'scatter' uses a scatter plot of the data points
# kind = 'reg' uses a regression plot(default order 1)
# kind = 'resid' uses a residual plot
# kind = 'kde' uses a kernel density estimate of the joint distribution
# kind = 'hex' uses a hexbin plot of the joint distribution
# Display the plot
plt.show()


# Print the first 5 rows of the DataFrame
print(auto.head())

# Plot the pairwise joint distributions from the DataFrame
sns.pairplot(auto)

# Display the plot
plt.show()


# Print the first 5 rows of the DataFrame
print(auto.head())

# Plot the pairwise joint distributions grouped by 'origin' along with regression lines
sns.pairplot(auto, kind='reg', hue='origin')

# Display the plot
plt.show()


# Print the covariance matrix
print(cov_matrix)

# Visualize the covariance matrix using a heatmap
sns.heatmap(cov_matrix)

# Display the heatmap
plt.show()




################### Time Series
# Import matplotlib.pyplot

# Plot the aapl time series in blue
plt.plot(aapl, color='blue', label='AAPL')

# Plot the ibm time series in green
plt.plot(ibm, color='green', label='IBM')

# Plot the csco time series in red
plt.plot(csco, color='red', label='CSCO')

# Plot the msft time series in magenta
plt.plot(msft, color='magenta', label='MSFT')

# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')

# Specify the orientation of the xticks
plt.xticks(rotation=60)

# Display the plot
plt.show()


# Plot the series in the top subplot in blue
plt.subplot(2, 1, 1)
plt.xticks(rotation=45)
plt.title('AAPL: 2001 to 2011')
plt.plot(aapl, color='blue')

# Slice aapl from '2007' to '2008' inclusive: view
view = aapl['2007':'2008']

# Plot the sliced data in the bottom subplot in black
plt.subplot(2, 1, 2)
plt.xticks(rotation=45)
plt.title('AAPL: 2007 to 2008')
plt.plot(view, color='black')
plt.tight_layout()
plt.show()


# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']

# Plot the sliced series in the top subplot in red
plt.subplot(2, 1, 1)
plt.xticks(rotation=45)
plt.title('AAPL: Nov. 2007 to Apr. 2008')
plt.plot(view, color='red', label='AAPL')

# Reassign the series by slicing the month January 2008
view = aapl['2008-01']

# Plot the sliced series in the bottom subplot in green
plt.subplot(2, 1, 2)
plt.xticks(rotation=45)
plt.title('AAPL: Jan. 2008')
plt.plot(view, color='green', label='AAPL')

# Improve spacing and display the plot
plt.tight_layout()
plt.show()


############### Plotting an inset view
# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']

# Plot the entire series
# plt.subplot(2,1,1)
plt.xticks(rotation=45)
plt.title('AAPL: 2001-2011')
plt.plot(aapl)

# Specify the axes
# plt.subplot(2,1,2)
plt.axes([0.25, 0.5, 0.35, 0.35])
# Plot the sliced series in red using the current axes
plt.plot(view, color='red')
plt.xticks(rotation=45)
plt.title('2007/11-2008/04')
plt.show()


# Plot the 30-day moving average in the top left subplot in green
plt.subplot(2, 2,1)
plt.plot(mean_30, 'green')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('30d averages')

# Plot the 75-day moving average in the top right subplot in red
plt.subplot(2, 2, 2)
plt.plot(mean_75, 'red')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('75d averages')

# Plot the 125-day moving average in the bottom left subplot in magenta
plt.subplot(2, 2, 3)
plt.plot(mean_125, 'magenta')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('125d averages')

# Plot the 250-day moving average in the bottom right subplot in cyan
plt.subplot(2, 2, 4)
plt.plot(mean_250, 'cyan')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('250d averages')



###################### histogram equalization of images
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Display image in top subplot using color map 'gray'
plt.subplot(2, 1, 1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image, cmap='gray')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2, 1, 2)
plt.xlim((0, 255))
plt.title('Normalized histogram')
plt.hist(pixels, bins=64, color='red', alpha=0.4, range=(0, 256), normed=True)

# Display the plot
plt.show()


# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Display image in top subplot using color map 'gray'
plt.subplot(2, 1, 1)
plt.imshow(image, cmap='gray')
plt.title('Original image')
plt.axis('off')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2, 1, 2)
pdf = plt.hist(pixels, bins=64, range=(0, 256), normed=False,
               color='red', alpha=0.4)
plt.grid('off')

# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()

# Display a cumulative histogram of the pixels
cdf = plt.hist(pixels, bins=64, range=(0, 256),
               cumulative=True, normed=True,
               color='blue', alpha=0.4)

# Specify x-axis range, hide axes, add title and display plot
plt.xlim((0, 256))
plt.grid('off')
plt.title('PDF & CDF (original image)')
plt.show()

##############################################################
##############################################################
################################################################
##############################################################
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')


file = os.listdir()[0]

image = plt.imread(file)

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Generate a cumulative histogram
cdf, bins, patches = plt.hist(pixels, bins=256, range=(
    0, 256), density=True, cumulative=True)
new_pixels = np.interp(pixels, bins[:-1], cdf*255)
# df_norm = (new_pixels - new_pixels.min()) / \
#     (new_pixels.max() - new_pixels.min())
# Reshape new_pixels as a 2-D array: new_image
# scaler = preprocessing.MinMaxScaler()
new_image = new_pixels.reshape(image.shape).astype(int)
# # new_pixels = scaler.fit_transform(new_pixels)
# new_image = df_norm.reshape(image.shape)

matplotlib.image.imsave('new_{}.png'.format(file[:-4]), new_image)

# Display the new image with 'gray' color map
# plt.subplot(2, 1, 1)
plt.title('Equalized image')
plt.axis('off')
plt.imshow(new_image)
plt.show()
# matplotlib.image.imsave('newImage.png', new_image)
# from PIL import Image
# im = Image.fromarray(new_image)
# im.save("New_{}.jpg".format(file[:-4]))


###########################################################################
########################################################################
# Generate a histogram of the new pixels
plt.subplot(2, 1, 2)
pdf = plt.hist(new_pixels, bins=64, range=(0, 256), normed=False,
               color='red', alpha=0.4)
plt.grid('off')

# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()
plt.xlim((0, 256))
plt.grid('off')

# Add title
plt.title('PDF & CDF (equalized image)')

# Generate a cumulative histogram of the new pixels
cdf = plt.hist(new_pixels, bins=64, range=(0, 256),
               cumulative=True, density=True,
               color='blue', alpha=0.4)
plt.show()


# Load the image into an array: image
image = plt.imread('hs-2004-32-b-small_web.jpg')

# Display image in top subplot
plt.subplot(2, 1, 1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image)

# Extract 2-D arrays of the RGB channels: red, blue, green
red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]

# Flatten the 2-D arrays of the RGB channels into 1-D
red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels = green.flatten()

# Overlay histograms of the pixels of each color in the bottom subplot
plt.subplot(2, 1, 2)
plt.title('Histograms from color image')
plt.xlim((0, 256))
plt.hist(red_pixels, bins=64, normed=True, color='red', alpha=0.2)
plt.hist(blue_pixels, bins=64, normed=True, color='blue', alpha=0.2)
plt.hist(green_pixels, bins=64, normed=True, color='green', alpha=0.2)

# Display the plot
plt.show()


# Load the image into an array: image
image = plt.imread('hs-2004-32-b-small_web.jpg')

# Extract RGB channels and flatten into 1-D array
red, blue, green = image[:,:,0], image[:,:,1], image[:,:,2]
red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels = green.flatten()

# Generate a 2-D histogram of the red and green pixels
plt.subplot(2,2,1)
plt.grid('off') 
plt.xticks(rotation=60)
plt.xlabel('red')
plt.ylabel('green')
plt.hist2d(x=red_pixels, y=green_pixels, bins=(32,32))

# Generate a 2-D histogram of the green and blue pixels
plt.subplot(2,2,2)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('green')
plt.ylabel('blue')
plt.hist2d(x=green_pixels, y=blue_pixels, bins=(32,32))

# Generate a 2-D histogram of the blue and red pixels
plt.subplot(2,2,3)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('blue')
plt.ylabel('red')
plt.hist2d(x=blue_pixels, y=red_pixels, bins=(32,32))

# Display the plot
plt.show()


