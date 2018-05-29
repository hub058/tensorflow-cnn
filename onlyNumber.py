import tensorflow as tf
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def random_captcha_text(char_set=number,char_size=4):
    captcha_text =[]
    for _ in range(char_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    # 把list转成字符串
    captcha_text = ''.join(captcha_text)
    
    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text+".png")
    captcha_image = Image.open(captcha,'r')
    captcha_image = np.array(captcha_image)
    return captcha_text,captcha_image

def convert2gray(img):
    if(len(img.shape)>2):
        gray = np.mean(img,-1)
        return gray
    else:
        return img

# 文本转向量
def text2vec(text):
    text_len = len(text)
    if(text_len > MAX_CAPTCHA):
        raise ValueError('验证码最长4个字符')
    # 初始化标签向量 4*10=40个的0值一维向量
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    for i,c in enumerate(text):
        idx = i*CHAR_SET_LEN + int(c)
        vector[idx] = 1
    return vector

# 向量转回文本
def vec2text(vec):
    text=[]
    # vec.nonzero:Returns a tuple of arrays (row,col)
    char_pos = vec.nonzero()[0]
    for _,c in enumerate(char_pos):
        number = c%10
        text.append(str(number))
        
    return "".join(text)


# 定义CNN网络
def crack_captcha_cnn(w_alpha=0.01,b_alpha=0.01):
    # tensorflow要求的格式是：batchsize,height,width,channel
    x = tf.reshape(X,shape= [-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])
    # 3层卷积网络(filter大小，输入维度，输出维度)
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1))
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1 = tf.nn.dropout(conv1,keep_prob)
    # 第二层神经网络的卷积层和池化层
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3,3,32,64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2))
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2 = tf.nn.dropout(conv2,keep_prob)
    # 第三层神经网络的卷积层和池化层
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3,3,64,64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3))
    conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv3 = tf.nn.dropout(conv3,keep_prob)
    # 全连接层 fully connected layer(希望转换为1024维的向量)
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64,1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1,w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense,keep_prob)
    # 第二层全连接层
    w_out = tf.Variable(w_alpha*tf.random_normal([1024,MAX_CAPTCHA*CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense,w_out),b_out)
    return out


def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size,IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size,MAX_CAPTCHA*CHAR_SET_LEN])
    # 有时生成的图像大小不是(60,160,3)
    def wrap_gen_captcha_text_and_image():
        # 这个死循环就是保证生成的图片符合要求，
        # 如果不符合要求就再去循环生成直到返回结果
        while True:
            text,image = gen_captcha_text_and_image()
            if image.shape ==(60,160,3):
                return text,image
            
    for i in range(batch_size):
        text,image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)
        batch_x[i,:] = image.flatten() / 255
        batch_y[i,:] = text2vec(text)
        
    return batch_x,batch_y


def train_crack_captcha_cnn():
    # crach:破解 captcha:验证码
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=output))
    optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1,MAX_CAPTCHA,CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict,2) # 预测的标签
    max_idx_l = tf.argmax(tf.reshape(Y,[-1,MAX_CAPTCHA,CHAR_SET_LEN]),2) # 真实的标签
    correct_pred= tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            # 训练
            batch_x,batch_y = get_next_batch(64)
            _, loss_ = sess.run([optm,loss],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            # print(step,loss_)
            
            # 每100step计算一次准确率
            if step%100 == 0:
                # 每100step预测一次
                batch_x_test ,batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy,feed_dict={X:batch_x_test,Y:batch_y_test,keep_prob:1.0})
                print(step,acc)
                # 如果准确率大于85%保存模型完成训练
                if(acc>0.85):
                    saver.save(sess,'./model/crack_captcha.model',global_step=step)
                    break
            step+=1


def crack_captcha(captcha_image):
    # 输出的是预测的40个标签每一个的概率
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/crack_capcha.model-810")
        predict = tf.argmax(tf.reshape(output, [-1,MAX_CAPTCHA,CHAR_SET_LEN]),2)
        text_list = sess.run(predict,feed_dict={X:[captcha_image],keep_prob:1.})
        return text_list[0].tolist()


if __name__=='__main__':
    train = 0
    if train==0:
        text, image = gen_captcha_text_and_image()
        print("验证码图像channel:",image.shape)
        # 图像的大小
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        MAX_CAPTCHA = len(text)
        char_set = number
        CHAR_SET_LEN = len(char_set)
        
        X = tf.placeholder(tf.float32, [None,IMAGE_HEIGHT*IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None,MAX_CAPTCHA*CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32) #drop
        
        train_crack_captcha_cnn()
        
    if train==1:
        # 使用训练好的模型
        number = ['0','1','2','3','4','5','6','7','8','9']
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        char_set = number
        CHAR_SET_LEN = len(char_set)
        text,image = gen_captcha_text_and_image()
        f = plt.figure()
        ax= f.add_subplot(111)
        ax.text(0.1,0.9,text,ha='center',va='center',transform=ax.transAxes)
        plt.imshow(image)
        plt.show()
        
        MAX_CAPTCHA = len(text)
        image = convert2gray(image)
        image = image.flatten()/255
        
        X = tf.placeholder(tf.float32,[None,IMAGE_HEIGHT*IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32,[None,MAX_CAPTCHA*CHAR_SET_LEN])
        keep_prob = tf.placeholder(tf.float32)
        
        predict_text = crack_captcha(image)
        print("正确:{}  预测:{}".format(text, predict_text))