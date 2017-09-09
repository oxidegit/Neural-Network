import numpy as np

class perceptron(object):
    def __init__(self, input_num, activation):
        self.W = np.array([0.0, 0.0])
        self.bias = 0.0
        self.activation = activation
    def train(self, input_vecs, labels, train_num, learning_rate):
        # wi = w-rate*(y-t)*xi
        # biax = rate(y-t)*xi

        length = self.W.shape[0]

        for i in range(train_num):
            print('\n第%d次训练， biax = %f' % (i, self.bias))
            print('w = ')
            print(self.W)

            for j in range(input_vecs.shape[1]):

                y = self.W.dot(input_vecs[:, j])
                print(y)
                y += self.bias
                y = self.activation(y)
                print(y)
                delta = labels[j]-y
                print ('delta%d'%delta)
                print('第%d次更新w开始' % (j))

                for n in range(length):

                    print ('reat = '+str(learning_rate*input_vecs[n][j]*delta))
                    print ('w[%d]yuanlai:'%(n)+str(self.W[n]))
                    self.W[n] = self.W[n]+learning_rate*input_vecs[n][j]*delta
                    print ('w[%d]zhihou'%(n)+str(self.W[n]))
                    self.bias = self.bias + learning_rate*delta
                print('\n第%d次更新完毕， biax = %f' % (j, self.bias))
                print('w = ')
                print(self.W)



    def predict(self, input_vec):
        y = self.W.dot(input_vec)
        y += self.bias
        y = self.activation(y)

        return y


if __name__ == '__main__':
    """dataLength = int(input('输入神经元个数：\n'))
    dataNum = int(input('数据量'))

    x = np.random.random((dataLength, dataNum))
    w = np.random.randint(0, 10, size=[1, dataLength])
    y = w.dot(x)"""
    input_vecs = [[1.,1.], [0.,0.], [1.,0.], [0.,1.]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1., 0., 1., 1.]

    p = perceptron(2,lambda x:1 if (x>0) else 0)
    print (np.array(input_vecs).transpose())
    p.train(np.array(input_vecs).transpose(), labels, 20, 0.1)
    print (p.predict([1, 0]))
    print(p.predict([1, 1]))
    print(p.predict([0, 1]))
    print(p.predict([0, 0]))

