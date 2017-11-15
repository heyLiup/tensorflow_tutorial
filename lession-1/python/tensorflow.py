import tensorflow as tf
import numpy as np
import pandas as pd

x = tf.placeholder(tf.string,shape=[None,])
image_bin = tf.decode_base64(x)
image_bin_reshape = tf.reshape(image_bin,shape=[-1,])
images = tf.map_fn(lambda img: tf.image.decode_png(img), image_bin_reshape,dtype=tf.uint8)



image_gray = tf.image.rgb_to_grayscale(images)
image_resized = tf.image.resize_images(image_gray, [48, 48],tf.image.ResizeMethod.NEAREST_NEIGHBOR)
image_resized_float = tf.image.convert_image_dtype(image_resized, tf.float32)


y1 = tf.placeholder(tf.float32,shape=[None, 24])
y2 = tf.placeholder(tf.float32,shape=[None, 24])
y3 = tf.placeholder(tf.float32,shape=[None, 24])
y4 = tf.placeholder(tf.float32,shape=[None, 24])


image_x = tf.reshape(image_resized_float,shape=[-1,48,48,1])
conv1 = tf.layers.conv2d(image_x, filters=32, kernel_size=[5, 5], padding='same')
norm1 = tf.layers.batch_normalization(conv1)
activation1 = tf.nn.relu(conv1)
pool1 = tf.layers.max_pooling2d(activation1, pool_size=[2, 2], strides=2, padding='same')
hidden1 = pool1

conv2 = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
norm2 = tf.layers.batch_normalization(conv2)
activation2 = tf.nn.relu(norm2)
pool2 = tf.layers.max_pooling2d(activation2, pool_size=[2, 2], strides=2, padding='same')
hidden2 = pool2


flatten = tf.reshape(hidden2, [-1, 12 * 12 * 64])


hidden3 = tf.layers.dense(flatten, units=1024, activation=tf.nn.relu)

letter1 = tf.layers.dense(hidden3, units=24)
letter2 = tf.layers.dense(hidden3, units=24)
letter3 = tf.layers.dense(hidden3, units=24)
letter4 = tf.layers.dense(hidden3, units=24)

letter1_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y1, logits=letter1))
letter2_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y2, logits=letter2))
letter3_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y3, logits=letter3))
letter4_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y4, logits=letter4))
loss = letter1_cross_entropy + letter2_cross_entropy+letter3_cross_entropy+letter4_cross_entropy

optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)

predict_concat = tf.stack([tf.argmax(letter1,1),
                           tf.argmax(letter2,1),
                           tf.argmax(letter3,1),
                           tf.argmax(letter4,1)],1)
y_concat = tf.stack([tf.argmax(y1,1),
                     tf.argmax(y2,1),
                     tf.argmax(y3,1),
                     tf.argmax(y4,1)],1)
accuracy_internal = tf.cast(tf.equal(predict_concat, y_concat),tf.float32),
accuracy = tf.reduce_mean(tf.reduce_min(accuracy_internal,2))

accuracy_letter =  tf.reduce_mean(tf.reshape(accuracy_internal,[-1]))

predict1 = tf.argmax(letter1,1)
predict2 = tf.argmax(letter2,1)
predict3 = tf.argmax(letter3,1)
predict4 = tf.argmax(letter4,1)

#生成最终结果
base_str = tf.constant("BCEFGHJKMPQRTVWXY2346789")
poses = tf.stack([predict1,predict2,predict3,predict4],axis=1)
length = tf.constant([1,1,1,1],tf.int64)
predicts = tf.map_fn(lambda pos: tf.substr(base_str,pos,length), poses, tf.string)
predict_join = tf.reduce_join(predicts,axis=1)

initer = tf.global_variables_initializer()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(initer)
saver.restore(sess)


pickup = "BCEFGHJKMPQRTVWXY2346789"
reader = pd.read_source(souce_id=802522,iterator=True)

identity = np.identity(24)
for i in range(15000):
  #获取图片数据
  df = reader.get_chunk(500)
  if df.empty:
    reader = pd.read_source(souce_id=802522,iterator=True)
    continue
  batch_y = df["code"].values
  batch_x = df["content"].values
  batch_y_1 = [identity[pickup.find(code[0])] for code in batch_y]
  batch_y_2 = [identity[pickup.find(code[1])] for code in batch_y]
  batch_y_3 = [identity[pickup.find(code[2])] for code in batch_y]
  batch_y_4 = [identity[pickup.find(code[3])] for code in batch_y]

  if i%10 == 0:
    print("step:"+str(i))
    accuracy_letter_,accuracy_ = sess.run([accuracy_letter,accuracy],feed_dict={x:batch_x,y1:batch_y_1,y2:batch_y_2,y3:batch_y_3,y4:batch_y_4})
    print(accuracy_letter_)
    print("accuracy is ====>%f"%accuracy_)
    if accuracy_ > 0.80:
      break
  sess.run(train_op,feed_dict={x:batch_x,y1:batch_y_1,y2:batch_y_2,y3:batch_y_3,y4:batch_y_4})

saver.save(sess)

prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({'image_base64': x},{'label': predict_join})

legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

builder = tf.saved_model.builder.SavedModelBuilder()
builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
           tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               prediction_signature,
      },
      legacy_init_op=legacy_init_op)
builder.save()
