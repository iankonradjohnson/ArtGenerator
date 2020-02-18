import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# set placeholders for variables used in train step

# images
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # dtype, [num images, size]

# weights
W = tf.Variable(tf.zeros([784, 10])) # matrix of 784 X 10 zeros

# biases
b = tf.Variable(tf.zeros([10])) # 10 biases for each number

# initialize variables
init = tf.initialize_all_variables()

# model
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)

# placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10]) # None is there because num images is not defined

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# training step
optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

# new session for training
sess = tf.Session()
sess.run(init)

for i in range(10000):
    # load batch of images 100 at a time
    batch_X, batch_Y = mnist.train.batch_X(100), mnist.train.batch_Y(100)
    train_data = {X: batch_X, Y_: batch_Y}
    
    # train
    sess.run(train_step, feed_dict = train_data)

    # success ? print it
    a,c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

    print(a,c)

    # success on test data?
    test_data = {X:mnist.test.images, Y_:mnist.test.labels}
    a,c = sess.run([accuracy, cross_entropy], feed_dict=test_data)

    print(a,c)



