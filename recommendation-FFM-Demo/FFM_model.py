import tensorflow as tf
import pandas as pd
import numpy as np
import os
'''
   原版代码解读：https://www.jianshu.com/p/781cde3d5f3d （好吧，估计这几个模型，全都是这一系列下对应的源码）
   
   在 FM基础上 引入了  将特征归属不同field的性质，这样  就相当于在之前FM式中加上了field
   field跟特征之间是  1：n的关系，也就是多个特征是属于同一个field的，我们在做特征交互时，也要考虑特征域所带来的隐向量作用关系



    看了下推荐系统遇上深度学习(二)--FFM模型理论和实践  就明白了，链接和这个代码是配套的。
    
    首先是（1） 生成数据集，指定 field匹配
          （2）定义在FFM公式中起主要作用的三个变量，三个权重项，首先是bias，然后是一维特征的权重，最后是交叉特征的权重。
          
                三者分别定义  模型方程中的   w0、wi   和最后相乘的那几个部分。可详见链接中的公式y(x)
                
                这里在最后一部分部分会牵扯到隐向量长度k，隐向量作用是针对每一维特征 xi，针对其它特征的每一种field fj，
                都会学习一个隐向量 v_i,fj， ，共有n*f个长度为k的隐变量，第三部分参数是牵扯到nfk的。
          （3）进行网络的训练阶段，使用输入 数据和  定义的权重进行FFM公式的计算，主要对应在inference的计算中。估计值的计算
          这里不能像FM一样先将公式化简再来做，对于交叉特征，只能写两重循环。【核心部分吧，对变量和输入参数的组织】
          （4）定义损失函数，并定义正则性，进行网络外部循环的训练。
          
    
    可以看出其核心就是 在FM公式基础上，加上了Field的概念，不同的特征可以归属同一个Field，这样进行进一步Field划分的原因是因为细化不同
    类型含义间特征的区别意义，   其相对于FM模型的特点是FM可以看作FFM的特例，是把所有特征都归属到一个field时的FFM模型。根据FFM的field敏感特性，
    可以导出其模型方程。
    
'''

#这里  定义有两个不用不同的field，  部分特征属于一种field，另一部分特征属于其他field.
#三个重要度维度的定义， 这用于进行权重的定义， 其中20是特征数量，  2是field的数量， 3是指 隐变量v的长度
input_x_size = 20
field_size = 2
vector_dimension = 3

total_plan_train_steps = 1000
# 使用SGD，每一个样本进行依次梯度下降，更新参数
batch_size = 1

all_data_size = 1000

lr = 0.01

MODEL_SAVE_PATH = "TFModel"
MODEL_NAME = "FFM"

def createTwoDimensionWeight(input_x_size,field_size,vector_dimension):
    #对 第三部分矩阵的定义，这个是最复杂的部分， 对应在公式中，隐变量长度是k个，FFM二次参数是nfk个， n表示特征数、f表示域数量、k表示假定的隐向量长度
    weights = tf.truncated_normal([input_x_size,field_size,vector_dimension])

    tf_weights = tf.Variable(weights)

    return tf_weights

def createOneDimensionWeight(input_x_size):
    weights = tf.truncated_normal([input_x_size])
    tf_weights = tf.Variable(weights)
    return tf_weights

def createZeroDimensionWeight():
    weights = tf.truncated_normal([1])
    tf_weights = tf.Variable(weights)
    return tf_weights

def inference(input_x,input_x_field,zeroWeights,oneDimWeights,thirdWeight):
    """计算回归模型输出的值
    这里是主模型结构
    """

    #公式中前两部分的计算
    secondValue = tf.reduce_sum(tf.multiply(oneDimWeights,input_x,name='secondValue'))

    firstTwoValue = tf.add(zeroWeights, secondValue, name="firstTwoValue")

    #公式第三部分的计算,使用一个双层的 循环获得结果
    thirdValue = tf.Variable(0.0,dtype=tf.float32) #第三部分累计和
    input_shape = input_x_size
    #可以严格按照公式去看，  外面的加是 i,内部循环是 i+1
    for i in range(input_shape):
        featureIndex1 = i
        fieldIndex1 = int(input_x_field[i])
        for j in range(i+1,input_shape):

            ########################第一部分  vi vj部分
            featureIndex2 = j
            fieldIndex2 = int(input_x_field[j])

            #Vi,f计算   注意看到公式中和下面计算上的区别，[featureIndex1,fieldIndex2,i]两个是反着的，表示不同的对应向量。
            vectorLeft = tf.convert_to_tensor([[featureIndex1,fieldIndex2,i] for i in range(vector_dimension)])
            weightLeft = tf.gather_nd(thirdWeight,vectorLeft) #允许在多维上进行索引,提取出矩阵。这是按照vectorLeft指定的index位置提取矩阵数据
            weightLeftAfterCut = tf.squeeze(weightLeft)#该函数返回一个张量，这个张量是将原始input中所有维度为1的那些维都删掉的结果
            # Vj,f计算
            vectorRight = tf.convert_to_tensor([[featureIndex2,fieldIndex1,i] for i in range(vector_dimension)])
            weightRight = tf.gather_nd(thirdWeight,vectorRight)
            weightRightAfterCut = tf.squeeze(weightRight)

            tempValue = tf.reduce_sum(tf.multiply(weightLeftAfterCut,weightRightAfterCut))

            #######################内部训练的 xi * xj 的部分    会获取不同的用于交叉的特征值。
            indices2 = [i]
            indices3 = [j]

            xi = tf.squeeze(tf.gather_nd(input_x, indices2))
            xj = tf.squeeze(tf.gather_nd(input_x, indices3))

            product = tf.reduce_sum(tf.multiply(xi, xj))

            secondItemVal = tf.multiply(tempValue, product)
            # 对第三部分结果的累加，thirdValue  是不断进行积累的。  assign就是一个赋值
            tf.assign(thirdValue, tf.add(thirdValue, secondItemVal))

    return tf.add(firstTwoValue,thirdValue)

def gen_data():
    '''
    这点其实有点不太好， 数据都是随机生成的，不太好捕捉 field所起到的不同feature作用
    '''
    labels = [-1,1]
    y = [np.random.choice(labels,1)[0] for _ in range(all_data_size)]
    x_field = [i // 10 for i in range(input_x_size)]
    x = np.random.randint(0,2,size=(all_data_size,input_x_size))

    print('生成的特征：',x.shape,x)
    print('随机生成的分类标签：',len(y),y)
    print('生成的每个特征所属的field,没有具体特征含义就可以这么任性：',x_field)
    return x,y,x_field


if __name__ == '__main__':
    global_step = tf.Variable(0,trainable=False)
    trainx,trainy,trainx_field = gen_data()
    #
    input_x = tf.placeholder(tf.float32,[input_x_size ])
    input_y = tf.placeholder(tf.float32)
    #

    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    ### 这三个Weights是用来干嘛的呢
    zeroWeights = createZeroDimensionWeight()

    oneDimWeights = createOneDimensionWeight(input_x_size)

    thirdWeight = createTwoDimensionWeight(input_x_size,  # 创建二次项的权重变量
                                           field_size,
                                           vector_dimension)  # n * f * k

    y_ = inference(input_x, trainx_field,zeroWeights,oneDimWeights,thirdWeight)



    l2_norm = tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.pow(oneDimWeights, 2)),
            tf.reduce_sum(tf.multiply(lambda_v, tf.pow(thirdWeight, 2)),axis=[1,2])
        )
    )
    #对输出结果附加 正则项，来防止模型过于复杂

    loss = tf.log(1 + tf.exp(input_y * y_)) + l2_norm

    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #共进行了1000轮对 FFM网络的训练。
        for i in range(total_plan_train_steps):
            for t in range(all_data_size):
                input_x_batch = trainx[t]
                input_y_batch = trainy[t]
                predict_loss,_, steps = sess.run([loss,train_step, global_step],
                                               feed_dict={input_x: input_x_batch, input_y: input_y_batch})

                print("After  {step} training   step(s)   ,   loss    on    training    batch   is  {predict_loss} "
                      .format(step=steps, predict_loss=predict_loss))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)
                writer = tf.summary.FileWriter(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), tf.get_default_graph())
                writer.close()
        #

















