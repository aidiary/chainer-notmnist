import chainer
import chainer.functions as F
import chainer.links as L


class LogReg(chainer.Chain):

    def __init__(self, n_out=10):
        super(LogReg, self).__init__(
            l1=L.Linear(None, n_out)
        )

    def __call__(self, x):
        # ここではlogitを返す
        # softmaxは、L.Classifier()のlossfunの
        # softmax_cross_entropy()でかかるのでここでは不要
        return self.l1(x)


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        self.activation = activation
        super(MLP, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_units),
            l4=L.Linear(None, n_units),
            l5=L.Linear(None, n_out)
        )

    def __call__(self, x):
        h = self.activation(self.l1(x))
        h = self.activation(self.l2(h))
        h = self.activation(self.l3(h))
        h = self.activation(self.l4(h))
        return self.l5(h)
