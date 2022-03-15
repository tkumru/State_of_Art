import tensorflow as tf

class Loss:
    def __init__(self, session, model) -> None:
        self.session = session
        self.model = model
    
    def main(self):
        sess = self.session
        model = self.model

        return self.content_loss(sess.run(model['conv4_2']), model['conv4_2'])

    def content_loss(self, p, x):
        """
        n = number of filters at layer 1
        m = height * width 
        """

        n = p.shape[3]
        m = p.shape[1] * p.shape[2]

        return (1 / (4 * n * m)) * tf.pow(x - p, 2)
