import tensorflow as tf

def tf_rotation6d_to_matrix(r6d):
    """ Compute rotation matrix from 6D rotation representation.
        Implementation base on 
            https://arxiv.org/abs/1812.07035
        [Inputs]
            6D rotation representation (last dimension is 6)
        [Returns]
            flattened rotation matrix (last dimension is 9)
    """
    tensor_shape = r6d.get_shape().as_list()

    if not tensor_shape[-1] == 6:
        raise AttributeError("The last demension of the inputs in tf_rotation6d_to_matrix should be 6, \
            but found tensor with shape {}".format(tensor_shape[-1]))

    with tf.compat.v1.variable_scope('rot6d_to_matrix'):
        r6d   = tf.reshape(r6d, [-1,6])
        x_raw = r6d[:,0:3]
        y_raw = r6d[:,3:6]
    
        x = tf.nn.l2_normalize(x_raw, axis=-1)
        z = tf.compat.v1.cross(x, y_raw)
        z = tf.nn.l2_normalize(z, axis=-1)
        y = tf.compat.v1.cross(z, x)

        x = tf.reshape(x, [-1,3,1])
        y = tf.reshape(y, [-1,3,1])
        z = tf.reshape(z, [-1,3,1])
        matrix = tf.concat([x,y,z], axis=-1)

        return matrix


def tf_matrix_to_rotation6d(mat):
    """ Get 6D rotation representation for rotation matrix.
        Implementation base on 
            https://arxiv.org/abs/1812.07035
        [Inputs]
            flattened rotation matrix (last dimension is 9)
        [Returns]
            6D rotation representation (last dimension is 6)
    """
    tensor_shape = mat.get_shape().as_list()

    if not ((tensor_shape[-1] == 3 and tensor_shape[-2] == 3) or (tensor_shape[-1] == 9)):
        raise AttributeError("The inputs in tf_matrix_to_rotation6d should be [...,9] or [...,3,3], \
            but found tensor with shape {}".format(tensor_shape[-1]))

    with tf.compat.v1.variable_scope('matrix_to_ration_6d'):
        mat = tf.reshape(mat, [-1, 3, 3])
        r6d = tf.concat([mat[...,0], mat[...,1]], axis=-1)

        if len(tensor_shape) == 1:
            r6d = tf.reshape(r6d, [6]) 
    return r6d