import numpy as np

# Create arrays of  ones, all integers

Vones = np.ones((4), dtype=np.int8)
Vones2 = np.ones((4, 6), dtype=np.int8)


# converting dtype
Vones3 = np.ones((9))
Vones3.dtype  # dtype('float64')
Vones3 = np.array(Vones3, dtype=np.int8)


# Create matrix of random integers
mat1 = np.random.randn(5, 5)
mat2 = np.random.randn(2, 3, 2)



# create character array
char1 = np.array([['Cup', 'Spoon', 'knives'], ['Table', 'Chairs', 'Beds']])
char1.dtype  # dtype('<U6')
print(char1)

char2 = np.array(['IP10mg', 'Placebo', 'IP10mg', 'Placebo', 'IP20mg'],
                 dtype='<U12')
char2.type
print(char2)




# Using tuples
tup1 = np.array([[(2, 3, 5), (9, 'man', 8)], [(9, 9, 4), ('Ken', 5, 3)]])
tup1.dtype
print(tup1)




# coping np array
tup2 = tup1  # this actually copy, but establish link to tup1
tup1[0, 0] = 'Josp'  # this will update values in tup2 also

# To actually copy, using copy() method
tup3 = tup1.copy()
tup1[0, 0] = 'Josphine'  # this wouldn't update values in tup3
