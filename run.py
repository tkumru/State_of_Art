import tensorflow as tf
from bin_art.image_process import Process
from bin_art.load_model import Load_VGG
from bin_art.style_loss import Style_Loss
from bin_art.loss import Loss
from bin_art.gif import GIF
import os

if __name__ == '__main__':
    OUTPUT_DIR = 'output/'
    STYLE_IMAGE = 'images/Çanakkale&Guernica/guernica.jpg'
    CONTENT_IMAGE = 'images/Çanakkale&Guernica/çanakkale.jpg'
    
    BETA = 5
    ALPHA = 100

    process = Process()
    process.resize_image(CONTENT_IMAGE, CONTENT_IMAGE)
    process.resize_image(STYLE_IMAGE, STYLE_IMAGE)

    with tf.compat.v1.Session() as sess:

        content_image = process.load_image(CONTENT_IMAGE)
        style_image = process.load_image(STYLE_IMAGE)

        model = Load_VGG().main()

        input_image = process.generate_noise_image(content_image=content_image)

        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(model['input'].assign(content_image))

        content_loss = Loss(session=sess, model=model).main()

        sess.run(model['input'].assign(style_image))

        style_loss = Style_Loss(session=sess, model=model).main()

        total_loss = BETA * content_loss + ALPHA * style_loss

        optimizer = tf.compat.v1.train.AdamOptimizer(2.0)
        train_step = optimizer.minimize(total_loss)

        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(model['input'].assign(input_image))

        iteration = 100
        for it in range(iteration):
            sess.run(train_step)

            if it % 100 == 0:
                mixed_image = sess.run(model['input'])

                print('Iteration %d' % (it))
                print('sum : ', sess.run(tf.reduce_sum(mixed_image)))
                print('cost: ', sess.run(total_loss))

                if not os.path.exists(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)

                filename = 'output/Çanakkale&Guernica/%d.png' % (it)
                process.save_image(filename, mixed_image)

    GIF(file_path="Çanakkale&Guernica", set_gifname="çanakkale").make_gif()
