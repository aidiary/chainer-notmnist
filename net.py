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
