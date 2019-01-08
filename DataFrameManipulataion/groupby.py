import pandas as pd 

############# groupby
# Group titanic by 'pclass'
by_class = titanic.groupby('pclass')

# Aggregate 'survived' column of by_class by count
# Number of those that survived by passengers class
count_by_class = by_class.survived.count()

# Print count_by_class
print(count_by_class)

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(['embarked', 'pclass'])

# Aggregate 'survived' column of by_mult by count
# Number of passengers that survived by embarked  and class
count_mult = by_mult.survived.count()

# Print count_mult
print(count_mult)
