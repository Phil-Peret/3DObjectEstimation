import numpy as np
import os
import pyvista as pv
import pymeshfix as fix
import numpy
import pickle
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
import rotation6d
import tensorflow as tf

#suppress warning tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def set_directory(start_dir):
    for dir in os.listdir(start_dir):
        path = os.path.join(start_dir,dir)
        if os.path.isdir(path):
            get_file_directory(path)

def get_file_directory(dir):
    print("File in" + dir)
    for file in os.listdir(dir):
        tree = os.path.join(dir,file)
        if os.path.isdir(tree):
            get_file_directory(tree)
        if os.path.isfile(tree):
            if file.endswith(".stl"):
                process_mesh(tree)


def get_processed_image(mesh_rotate, pl, mean_curvature, gauss_curvature):
    #different position camera
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
        pl.remove_actor(mesh_actor)
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
            #plt.imshow(image)
            #plt.show()
            images.append(image)
    #for image in images:
        #plt.imshow(image)#[:,:,1],cmap='gray')
        #plt.show()
    #concat_image=concatenate_image(images)
    return images


def process_mesh(path_file):
    stl_mesh = load_mesh(path_file)
    #stl_mesh.plot()
    stl_mesh = stl_mesh.extract_geometry()
    for i in range(300):
        mesh_rotate, quat_rotation = random_rotation(stl_mesh)
        pl = pv.Plotter(off_screen=True,lighting='none')
        #mean curvature
        mean_curvature = mesh_rotate.curvature('mean')
        mean_curvature = filter_value(mean_curvature)
        #gauss curvature
        gauss_curvature = mesh_rotate.curvature('gaussian')
        gauss_curvature = filter_value(gauss_curvature)
        images=get_processed_image(mesh_rotate, pl, mean_curvature, gauss_curvature)
        pl.close()
        pl.deep_clean()
        data.append((images, quat_rotation))
 

def show_mesh(mesh):
    pl = pv.Plotter()
    pl.add_mesh(mesh)
    pl.camera_position = 'yz'
    pl.show()
    pl.close()
    pl.deep_clean()

def save_object(obj):
    try:
        with open('test_wo_z.pickle', 'wb') as f:
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
set_directory('models')
save_object(data)
