import tensorflow as tf
import tensorflow.contrib.layers as layers


def conv_layer(name, input, ksize, depth, kstep, padding='SAME', activate_fun=tf.nn.relu):

    '''
    实现一层卷积神经网络的卷积层的前向传播
    :param name: 该层卷积网络的名称
    :param input: 该层卷积网络的输入矩阵
    :param ksize: 该层卷积网络的卷积核大小
    :param depth: 该层卷积网络的卷积核深度
    :param kstep: 该层卷积网络卷积核移动的步长
    :param padding: 该层卷积网络的输入矩阵是否填充0,‘SAME’为填充0, ‘VAILD’ 为不填充0
    :param activate_fun: 该层卷积的激活函数
    :return: 返回卷积后的矩阵数据
    '''

    if kstep is None:
        kstep = [1, 1]
    chanel = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope+'w',
                                  shape=[ksize[0],ksize[1],chanel,depth],
                                  dtype=tf.float32,
                                  initializer=layers.xavier_initializer_conv2d())
        biases = tf.get_variable(scope+'b',
                                 shape=[depth],
                                 dtype=tf.float32,
                                 initializer=tf.zeros_initializer())
        conv_results = tf.nn.conv2d(input,
                                    weights,
                                    strides=[1,kstep[0],kstep[1],1],
                                    padding=padding)
        biases_results = tf.nn.bias_add(conv_results,biases)
        activate_results = biases_results
        if activate_fun != None:
            activate_results = activate_fun(biases_results, name=scope)
        return activate_results


def pool_layer(name, input, ksize, kstep, padding='SAME', pool_fun=tf.nn.max_pool):

    '''
    实现卷积神经网络的一层池化层的前向传播
    :param name: 该池化层的名称
    :param input: 该池化层输入的矩阵
    :param ksize: 该池化层输入的卷积核大小
    :param kstep: 该池化层卷积核移动的步长
    :param padding: 该池化层是否对输入矩阵进行0填充，‘SAME’为进行0填充，‘VAILD’为不进行0填充
    :param pool_fun: 该池化层所使用的池化函数
    :return: 返回池化层输出的池化作用后的矩阵数据
    '''

    with tf.name_scope(name) as scope:
        pool_results = pool_fun(input,
                                ksize=[1,ksize[0],ksize[1],1],
                                strides=[1,kstep[0],kstep[1],1],
                                padding= padding,
                                name=scope)
        return pool_results


def convert_n_to_2(name, input):

    '''
    将输入的矩阵数据拉成一维数据
    :param name: 转换操作的名称
    :param input: 输入的矩阵，维度大于2
    :return: 返回一个二维数据[batch,node]
    '''

    shape = input.get_shape()
    convert_shape = shape[1].value
    for i in range(2,len(shape)):
        convert_shape *= shape[i].value
    results = tf.reshape(input, [-1, convert_shape], name=name)
    return results


def fc_layer(name, input, output_size, activate_fun=tf.nn.relu):

    '''
    实现一层全连接网络的前向传播
    :param name: 该层全连接网络的名称
    :param input: 该层全连接网络的输入矩阵
    :param output_size: 该层全连接网络的输出节点个数
    :param activate_fun: 该层全链接网络使用的激活函数
    :return: 返回该层全连接网络的输出矩阵数据
    '''

    # input有可能为一个二维矩阵，即[batch,input_node],因此这里取-1而不是取0
    input_size = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope+"w",
                                  shape=[input_size,output_size],
                                  dtype=tf.float32,
                                  initializer=layers.xavier_initializer_conv2d())
        biases = tf.get_variable(scope+"b",
                                 shape=[output_size],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        output_results = tf.add(tf.matmul(input,weights),biases)
        activate_results = output_results
        if activate_fun != None:
            activate_results = activate_fun(output_results,name=scope)
        return activate_results


def rnn_cell(name, input, state, activate_fun=tf.nn.tanh):

    '''
    实现循环神经网络的一个序列节点的一层前向传播
    :param name: 该层循环网络的名称
    :param input: 该层循环网络输入序列的一个子序列节点
    :param state: 该层循环网络的上一个状态
    :param activate_fun:该层循环网络的激活函数
    :return: 返回该层循环网络的输出矩阵及状态值
    '''

    input_size = input.get_shape()[-1].value
    state_size = state.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights_input = tf.get_variable(scope+"input_w",
                                        dtype=tf.float32,
                                        shape=[input_size,state_size],
                                        initializer=tf.constant_initializer(0.1))
        weights_state = tf.get_variable(scope+"state_w",
                                        shape=[state_size,state_size],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.2))
        biases = tf.get_variable(scope+"b",
                                 shape=[state_size],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.3))
        outputs_rnn = tf.matmul(input_size,weights_input) + \
            tf.matmul(state,weights_state) + biases
        activate_results = outputs_rnn
        current_state = activate_results
        if activate_fun != None:
            activate_results = activate_fun(outputs_rnn,name=scope)
            current_state = activate_results
        return activate_results,current_state


