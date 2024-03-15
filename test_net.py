import numpy as np
import os
import pyvista as pv
import pymeshfix as fix
import matplotlib.pyplot as plt
import numpy
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import keras as k
import tensorflow as tf
import rotation6d

def get_image_by_mesh(mesh, camera_position='xz'):
    pl = pv.Plotter(off_screen=True)
    with pl.window_size_context((100, 100)):
        pl.add_mesh(mesh)
        pl.camera_position=camera_position
        pl.show()
        image = pl.screenshot(window_size=(100,100), return_img=True)
        pl.close()
        #image = pl.screenshot(window_size=(200,200), return_img=True)
        return image

def show_mesh_diffpos(mesh):
    position_camera=['xy', 'xz', 'yz', 'yx', 'zx', 'zy']
    for position in position_camera:
        pl = pv.Plotter()
        pl.add_mesh(mesh)
        pl.camera_position = position
        pl.show()
    pl.close()
    pl.deep_clean()
    del pl


def min_angle_error(R1, R2):
    diff = np.trace(np.dot(np.transpose(R1), R2))
    # Calcola l'angolo di rotazione minimo
    angle = np.arccos((diff - 1.0) / 2.0)
    angle_degree = np.degrees(angle)
    return angle_degree

def geodistic_distance_matrix_loss(y_true, y_pred):
    y_pred_m=tf.reshape(rotation6d.tf_rotation6d_to_matrix(y_pred), [-1,3,3])
    y_true_m=tf.reshape(rotation6d.tf_rotation6d_to_matrix(y_true), [-1,3,3])
    y_true_m_transposed = tf.transpose(y_true_m, perm=[0, 2, 1])

    # Calcolo della traccia della differenza tra le matrici
    trace_diff = tf.linalg.trace(tf.matmul(y_true_m_transposed, y_pred_m))

    # Clipping
    trace_diff = tf.clip_by_value(trace_diff, -1.0, 3.0)

    # Calcolo dell'angolo differenziale
    angle_diff = tf.math.acos(tf.clip_by_value(0.5 * (trace_diff - 1.0), -1.0, 1.0))

    # Calcolo della distanza geodetica media nel batch
    loss = tf.reduce_mean(angle_diff)
    return loss

def set_directory(start_dir):
    for dir in os.listdir(start_dir):
        path = os.path.join(start_dir,dir)
        if os.path.isdir(path):
            get_file_directory(path)

def get_file_directory(dir):
    print("File in " + dir)
    for file in os.listdir(dir):
        tree = os.path.join(dir,file)
        if os.path.isdir(tree):
            get_file_directory(tree)
        if os.path.isfile(tree):
            if file.endswith(".stl"):
                process_mesh(tree)

def show_image(image):
    plt.imshow(image)
    plt.show()

def get_processed_image(mesh_rotate, pl, mean_curvature, gauss_curvature):
    position_camera=['xy', 'xz', 'yz', 'yx', 'zx', 'zy']
    images=[]
    for position in position_camera:
        mesh_rotate['Data'] = mean_curvature
        mesh_actor=pl.add_mesh(mesh_rotate, cmap='gray')
        pl.camera_position = position
        pl.hide_axes()
        pl.remove_scalar_bar()
        image = pl.screenshot(window_size=(100,100), return_img=True)
        #remove current mesh
        pl.remove_actor(mesh_actor)
        #add value gauss curvature mesh
        mesh_rotate['Data'] = gauss_curvature
        #add mesh
        mesh_actor = pl.add_mesh(mesh_rotate, cmap='gray')
        pl.hide_axes()
        pl.remove_scalar_bar()
        image_gauss = pl.screenshot(window_size=(100,100), return_img=True)
        #pl.remove_actor(mesh_actor)
        #replace channel G with mean curvature
        image[:,:,1] = image_gauss[:,:,0]
        #plt.imshow(image)
        #plt.show()
        #depth image
        with pl.window_size_context((100, 100)):
            mesh_actor = pl.add_mesh(mesh_rotate)
            pl.hide_axes()
            pl.remove_scalar_bar()
            image_depth=pl.get_image_depth()
            image_depth=numpy.asmatrix(image_depth)
            image_depth[np.isnan(image_depth)]=np.nanmin(image_depth)
            min_img, max_img = image_depth.min(), image_depth.max()
            image_depth = (image_depth - min_img)/(max_img - min_img)
            image_depth = image_depth*255
            image_depth = image_depth.astype(int)
            #replace B channel with depth value (0:255)
            image[:,:,2]=image_depth
            #plt.imshow(image[:,:,1],cmap='gray')
            images.append(image)
    #for image in images:
        #plt.imshow(image)#[:,:,1],cmap='gray')
        #plt.show()
    #concat_image=concatenate_image(images)
    return images

def test_model(images, rotation, mesh_rot, mesh_origin, data_predict):
    compare_image=[]
    mesh_cp = mesh_rot.copy()
    model = tf.keras.models.load_model('trained_orientation6d.keras', custom_objects={'geodistic_distance_matrix_loss': geodistic_distance_matrix_loss})
    result = model.predict([np.array([images[0]]), np.array([images[1]]), np.array([images[2]]), \
                            np.array([images[3]]), np.array([images[4]]), np.array([images[5]])])
    result_matrix = rotation6d.tf_rotation6d_to_matrix(tf.convert_to_tensor(result))
    result_matrix = (result_matrix.numpy())
    inv_result_matrix = R.from_matrix(result_matrix).inv().as_matrix()
    matrix_transform = numpy.identity(4)
    matrix_transform[0:3,0:3] = inv_result_matrix
    #q_diff=quaternion_difference(np.quaternion(*result),np.quaternion(*rotation))
    #rotazione inversa
    mesh_cp.transform(matrix_transform)
    compare_image.append((get_image_by_mesh(mesh_rot),get_image_by_mesh(mesh_cp)))
    #plt.imshow(compare_image[0][0])
    #plt.show()
    #plt.imshow(compare_image[0][1])
    #plt.show()
    data_predict.append((compare_image))

def plot_data_predict(data_predict, mesh):
    X=np.arange(0,len(data_predict),1)
    mesh_images=np.array([data[0] for data in data_predict])
    
    figure, axis = plt.subplots(6,6)
    plt.subplots_adjust(left=0.035, right=0.975, bottom=0.035, top=0.975, wspace=0.1, hspace=0.25)
    row,col = axis.shape
    index=0
    for i in range(row-1):
        for j in range(col):
            if j%2==0:
                axis[i,j].imshow(mesh_images[index][0].reshape((100,100,3)))
                if i==0:
                    axis[i,j].set_title('mesh')
            else:
                axis[i,j].imshow(mesh_images[index][1].reshape((100,100,3)))
                if i==0:
                    axis[i,j].set_title('predicted rotation')
                index+=1
    plt.show()

def process_mesh(path_file):
    data_predict=[]
    stl_mesh = load_mesh(path_file)
    show_mesh(stl_mesh)
    #show_mesh_diffpos(stl_mesh)
    stl_mesh = stl_mesh.extract_geometry()
    for i in range(15):
        mesh_rotate, quat_rotation = random_rotation(stl_mesh)
        pl = pv.Plotter(off_screen=True,lighting='none')
        mesh_rotate_wo_data = mesh_rotate.copy()
        #mean curvature
        mean_curvature = mesh_rotate.curvature('mean')
        mean_curvature = filter_value(mean_curvature)
        #gauss curvature
        gauss_curvature = mesh_rotate.curvature('gaussian')
        gauss_curvature = filter_value(gauss_curvature)
        images=get_processed_image(mesh_rotate, pl, mean_curvature, gauss_curvature)
        pl.close()
        pl.deep_clean()
        del pl
        test_model(images, quat_rotation, mesh_rotate_wo_data, stl_mesh, data_predict)
    plot_data_predict(data_predict, stl_mesh)
    
        
def show_mesh(mesh):
    pl = pv.Plotter()
    pl.add_mesh(mesh, cmap='gray')
    pl.show()
    pl.close()
    pl.deep_clean()
    del pl

def save_object(obj):
    try:
        with open('data.pickle', 'wb') as f:
            pickle.dump(obj, f)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def random_rotation(mesh):
    #show_mesh(mesh)
    rotated_mesh = mesh.copy()
    euler_rotation = R.random().as_euler('zxy')
    #print(quaternion_rotation)
    matrix_transform = numpy.identity(4)
    matrix_rotation = R.from_euler('zxy',euler_rotation).as_matrix()
    matrix_transform[0:3,0:3] = matrix_rotation
    #mesh_euler = mesh.copy()
    rotated_mesh.transform(matrix_transform)
    euler_rotation[0]=0
    matrix_rotation_wo_z = R.from_euler('zxy',euler_rotation).as_matrix()
    rot6d=rotation6d.tf_matrix_to_rotation6d(tf.convert_to_tensor(matrix_rotation_wo_z))
    #print(quaternion_rotation)
    #show_mesh(mesh)
    #show_mesh(rotated_mesh)
    return rotated_mesh, rot6d

def filter_value(arr):
    filter_array=[]
    for element in arr:
        if element >=2:
            filter_array.append(2)
        elif element <=-2:
            filter_array.append(-2)
        else:
            filter_array.append(element)
    filter_array=np.asarray(filter_array)
    return filter_array

def load_mesh(path):
    print('Mesh path = '+path)
    global manifold_parts
    #--clean mesh
    pv_mesh=pv.read(path)
    if pv_mesh.is_manifold:
        arr_mesh=pv_mesh.split_bodies()
        manifold_parts=int(len(arr_mesh))
        return arr_mesh
    else:
        pv_mesh=pv_mesh.clean()
        pv_mesh.fill_holes(10)
        #--split mesh
        arr_mesh=pv_mesh.split_bodies()
        manifold_parts=0
        for i,x in enumerate(arr_mesh):
            mesh_x=x.extract_surface()
            mesh_fix=fix.MeshFix(mesh_x)
            if(mesh_x.is_manifold): 
                manifold_parts+=1
            else:
                mesh_fix.repair(joincomp=True,remove_smallest_components=False)
                res=pv.wrap(mesh_fix.mesh)
                res=res.extract_surface()
                arr_mesh[i]=res
        return arr_mesh

print(os.listdir())
data=[]
'''dir models, start automatically process mesh'''
set_directory('test')
save_object(data)
