import tensorflow as tf

class Style_Loss:
    def __init__(self, session, model) -> None:
        self.session = session
        self.model = model

        self.style_layers = [
            ('conv1_1', 0.5),
            ('conv2_1', 1.0),
            ('conv3_1', 1.5),
            ('conv4_1', 3.0),
            ('conv5_1', 4.0),
        ]

    def main(self):
        sess = self.session
        model = self.model
        
        e = [self._style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in self.style_layers]
        w = [w for _, w in self.style_layers]
        loss = sum([w[l] * e[l] for l in range(len(self.style_layers))])

        return loss

    def gram_matrix(self, f, n, m):
        ft = tf.reshape(f, (m, n))

        return tf.matmul(tf.transpose(ft), ft)

    def _style_loss(self, a, x):
        n = a.shape[3]
        m = a.shape[1] * a.shape[2]
        A = self.gram_matrix(a, n, m)
        G = self.gram_matrix(x, n, m)

        return (1 / (4 * n**2 * m**2)) * tf.reduce_sum(tf.pow(G - A, 2))
