""" import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test) """


# from __future__ import print_function

import tensorflow as tf
try:
  tf.contrib.eager.enable_eager_execution()
  print("TF imported with eager execution!")
except ValueError:
  print("TF already imported with eager execution!")

# try:
#   tf.contrib.eager.enable_eager_execution()
# except ValueError:
#   pass  # enable_eager_execution errors after its first call

tensor = tf.constant('Hello, world!')
tensor_value = tensor.numpy()
print(tensor)
print(tensor_value)


# Create a graph.
g = tf.Graph()

# Establish our graph as the "default" graph.
with g.as_default():
  # Assemble a graph consisting of three operations. 
  # (Creating a tensor is an operation.)
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  my_sum = tf.add(x, y, name="x_y_sum")
  
  # Task 1: Define a third scalar integer constant z.
  z = tf.constant(4, name="z_const")
  # Task 2: Add z to `my_sum` to yield a new sum.
  new_sum = tf.add(my_sum, z, name="x_y_z_sum")

  # Now create a session.
  # The session will run the default graph.
  with tf.Session() as sess:
    # Task 3: Ensure the program yields the correct grand total.
    print(new_sum.eval())
    print(tensor_value) 

v = tf.contrib.eager.Variable([[1, 2, 3], [4, 5, 6]])
print(v.numpy())

try:
  print("Assigning [7, 8, 9] to v")
  v.assign([7, 8, 9])
except ValueError as e:
  print("Exception:", e)


die1 = tf.contrib.eager.Variable(
    tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
die2 = tf.contrib.eager.Variable(
    tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))

dice_sum = tf.add(die1, die2)
resulting_matrix = tf.concat(values=[die1, die2, dice_sum], axis=1)

print(resulting_matrix.numpy())
my_variable = tf.get_variable("my_variable", [1, 2, 3])
mybb = tf.get_variable("mybb", [1, 2, 3])
vart = tf.get_variable("vart", [1, 2, 3])
mble = tf.get_variable("mble", [1, 2, 3])


print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
