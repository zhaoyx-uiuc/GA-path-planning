

import numpy as np
from numpy import dot
from math import sqrt
import open3d as o3d
import bpy
import bmesh
from scipy.stats import special_ortho_group
import torch
def greedy_minimize_rows(matrix):
    

    selected_rows=[]
    selected_row_indices = []
    num_depth, num_rows, num_cols = matrix.shape
    uncovered_cols = np.array(range(num_cols))
    
    while uncovered_cols.shape[0]!=0:
        depth_row_sums = np.sum(matrix[:, :, uncovered_cols], axis=2)
        row_with_most_ones = np.array(np.unravel_index(np.argmax(depth_row_sums), depth_row_sums.shape))
        selected_row = matrix[row_with_most_ones[0],row_with_most_ones[1], :]
        selected_rows.append(selected_row)
        selected_row_indices.append(row_with_most_ones)

        indices_to_keep = np.where(selected_row == 0)[0]
        uncovered_cols=uncovered_cols[np.isin(uncovered_cols,indices_to_keep)]
        #uncovered_cols=np.array(new_uncovered_cols)
    selected_row_indices_array=np.array(selected_row_indices)
    selected_matrix = matrix[selected_row_indices_array[:,0],selected_row_indices_array[:,1], :]

    return selected_row_indices_array, selected_matrix

def is_inside_polygon(polygon, point):
    # the input polygon vertices must be in sequence
    winding_number = 0

    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]

        if y1 <= point[1]:
            if y2 > point[1] and (x2 - x1) * (point[1] - y1) > (y2 - y1) * (point[0] - x1):
                winding_number += 1
        else:
            if y2 <= point[1] and (x2 - x1) * (point[1] - y1) < (y2 - y1) * (point[0] - x1):
                winding_number -= 1

    return winding_number!=0
def signedVolume(a, b, c, d):
    """Computes the signed volume of a series of tetrahedrons defined by the vertices in 
    a, b c and d. The ouput is an SxT array which gives the signed volume of the tetrahedron defined
    by the line segment 's' and two vertices of the triangle 't'."""

    return np.sum((a-d)*np.cross(b-d, c-d), axis=2)

def segmentsIntersectTriangles(s, t):
    """For each line segment in 's', this function computes whether it intersects any of the triangles
    given in 't'."""
    # compute the normals to each triangle
    normals = np.cross(t[2]-t[0], t[2]-t[1])
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # get sign of each segment endpoint, if the sign changes then we know this segment crosses the
    # plane which contains a triangle. If the value is zero the endpoint of the segment lies on the 
    # plane.
    # s[i][:, np.newaxis] - t[j] -> S x T x 3 array
    sign1 = np.sign(np.sum(normals*(s[0][:, np.newaxis] - t[2]), axis=2)) # S x T
    sign2 = np.sign(np.sum(normals*(s[1][:, np.newaxis] - t[2]), axis=2)) # S x T

    # determine segments which cross the plane of a triangle. 1 if the sign of the end points of s is 
    # different AND one of end points of s is not a vertex of t
    cross = (sign1 != sign2)*(sign1 != 0)*(sign2 != 0) # S x T 

    # get signed volumes
    v1 = np.sign(signedVolume(t[0], t[1], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    v2 = np.sign(signedVolume(t[1], t[2], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    v3 = np.sign(signedVolume(t[2], t[0], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T

    same_volume = np.logical_and((v1 == v2), (v2 == v3)) # 1 if s and t have same sign in v1, v2 and v3

    return (np.sum(cross*same_volume, axis=1) > 0)

#import cupy as cp
def find_pairs(array1, array2):


    # Find the matching elements and their indices
    matches = np.where(array1[..., np.newaxis] == array2)

    # Get the row indices of array1 for the matching elements
    row_indices = matches[0]

    # Get the indices of array2 for the matching elements
    col_indices = matches[2]

    # Combine the row indices and column indices into pairs
    pairs = np.column_stack((row_indices.flatten(), col_indices.flatten()))

    return pairs
def waypoint_to_center(waypoint,center): 
    direction=center-waypoint
    norm=np.linalg.norm(direction)
    direction=direction/norm
    return direction

def output_visualizable_faces_and_matrix(model_path,vis_arr,save_path,range):
    # range is a ndarray(4,2) representing 4 2D points.
    for obj in bpy.context.scene.objects:
        bpy.data.objects.remove(obj)

# Restore default settings
    bpy.ops.wm.read_factory_settings()

    bpy.ops.import_scene.obj(filepath=model_path)
    idx=0
    idx_list=[]
    for obj in bpy.data.objects:
        if obj.type == 'MESH'and obj.name!='Cube' and obj.name!='Cube.001':
            bpy.ops.object.mode_set(mode='EDIT')
            bm=bmesh.from_edit_mesh(obj.data)
            bm.faces.ensure_lookup_table()
            for face in bm.faces:
                center=np.asarray(face.calc_center_median())               
                if vis_arr[idx]==1 and is_inside_polygon(range,center):
                    face.select=True
                    idx_list.append(idx)
                else:
                    bm.faces.ensure_lookup_table()
                    bmesh.ops.delete(bm,geom=[face],context='FACES')                
                idx+=1
    bpy.ops.export_scene.obj(filepath=save_path+'visualizable.obj')
    return np.asarray(idx_list)

def output_visualizable_faces(model_path,vis_arr,save_path):
    for obj in bpy.context.scene.objects:
        bpy.data.objects.remove(obj)

# Restore default settings
    bpy.ops.wm.read_factory_settings()

    bpy.ops.import_scene.obj(filepath=model_path)
    idx=0
    for obj in bpy.data.objects:
        if obj.type == 'MESH'and obj.name!='Cube' and obj.name!='Cube.001':
            bpy.ops.object.mode_set(mode='EDIT')
            bm=bmesh.from_edit_mesh(obj.data)
            bm.faces.ensure_lookup_table()
            for face in bm.faces:               
                if vis_arr[idx]==1:
                    face.select=True
                else:
                    bm.faces.ensure_lookup_table()
                    bmesh.ops.delete(bm,geom=[face],context='FACES')                
                idx+=1
    bpy.ops.export_scene.obj(filepath=save_path+'visualizable.obj')

def random_rotate(v, kappa):
    # Generate a random rotation matrix from the SO(3) group
    R = special_ortho_group.rvs(3)

    # Generate a random vector from the von Mises-Fisher distribution
    random_vector = np.random.randn(3)
    random_vector /= np.linalg.norm(random_vector)
    rotated_vector = np.dot(R, random_vector)

    # Adjust the rotation to be biased towards the original direction
    rotated_vector = kappa * v + (1 - kappa) * rotated_vector
    rotated_vector /= np.linalg.norm(rotated_vector)

    return rotated_vector

def angle_judgement(arr1,arr2,angle_thres_cos):
    #arr1_cp=cp.asarray(arr1)
    #arr2_cp=cp.asarray(arr2)
    is_in_angle=abs(np.dot(arr1,arr2))>angle_thres_cos
    return is_in_angle
def Write3dcoord_topath(point_array,name):
# Define the line segments as a list of tuples
    line_coords = []
    for idx in range(point_array.shape[0]-1):
        line_coords.append((idx,idx+1))
    

# Create an Open3D PointCloud object from the numpy array
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(point_array)

# Create an Open3D LineSet object from the line segment tuples
    lines = o3d.geometry.LineSet()
    lines.points = points.points
    lines.lines = o3d.utility.Vector2iVector(line_coords)

    # Write the PointCloud and LineSet to a PLY file
    o3d.io.write_line_set(name, lines)

def Write3dcoord_toply(point_array,name):
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(point_array)
    o3d.io.write_point_cloud(name+'.ply',pcd)
def calculate_coverage(positions,sights,vis_mat,dir_mat,lin_parameters,area_arr,angle_cos_thres):
    """
    calculate the coverage of a path
    Parameters:
        positions - NDArray(n_pathviewpoints,3)  The coordinates of viewpoints on the path
        sights - NDArray(n_pathviewpoints,3)  The poses of viewpoints on the path. All the vectors are unit vectors(normal=1). 
        vis_mat - NDArray(n_faces,n_viewpoints)  Visibility matrix indicating if a face is visible from a viewpoint.
        dir_mat - NDArray(n_viewpoints, n_faces, 3) The vectors from each viewpoint to each face. All the vectors are unit vectors.
        lin_parameters - (xmin,xmax,ymin,ymax,zmin,zmax,step)  The boundary and grid size information
        area_arr - NDArray(n_faces, 1)  The area of faces 
    Returns:
        coverage - NDArray(1,1)  The coverage of the path
        vis_arr - NDArray(n_faces, 1)  A vector indicating which faces are visible from the path
    """
    path_vis_noFOV_matrix = torch.zeros((0, vis_mat.shape[0]), dtype=torch.float32)
    
    idx_array = []
    for position in positions:
        wp_idx = cal_idx_from_coord(position, lin_parameters)
        idx_array.append(wp_idx)
    idx_array = torch.tensor(idx_array, dtype=torch.int32)
    path_vis_noFOV_matrix = vis_mat[:, idx_array]
    path_vis_noFOV_matrix = path_vis_noFOV_matrix.T
    if sights is not None:
    #path_dir_matrix = torch.zeros((0, vis_mat.shape[0], 3), dtype=torch.float32)
        path_dir_matrix = dir_mat[idx_array, :, :]
        cos_angle_matrix = torch.einsum('ij,ikj->ik', sights, path_dir_matrix)
        within_angle_matrix = torch.where(cos_angle_matrix > angle_cos_thres, torch.tensor(1), torch.tensor(0))
        path_vis_matrix = within_angle_matrix * path_vis_noFOV_matrix
    else:
        path_vis_matrix=path_vis_noFOV_matrix
    path_vis_array = torch.sum(path_vis_matrix, dim=0)
    path_vis_array = torch.where(path_vis_array > 0, torch.tensor(1), torch.tensor(0))

    coverage = torch.dot(path_vis_array.float(), area_arr) / torch.sum(area_arr)
    return coverage, path_vis_array

def getPathCoverage(consider_self_hit,model_path,view_points,cam_lens,directions,FOV,FOV_limit,dist_thres,angle_thres,vis_path,ground_faces):
    """
    Calculate the coverage of a set of viewpoints on a model
    Parameters:
        model_path - String  The file path of the model. The format should be obj
        view_points - NDArray(number of viewpoints, 3)  The position of the viewpoints.
        cam_lens - NDArray(number of viewpoints)  The lens of camera on each viewpoint
        rotations - NDArray(number of viewpoints, 3)  The euler rotation angle in radius along x,y,z axis of each viewpoint
        FOV - Float  The field of view (half-sight) in radius
        FOV_limit - Bool  If FOV limit is applied
        dist_thres - Float  The distance threshold for judging if a face is visible. If the distance between the face and the
        viewpoint is larger than the threshold, then the face will be regarded as invisible
        angle_thres - Float  The inclination angle threshold for judging if a face is visible. If the inclination angle
        between the normal of the face and the ray direction is larger than the threshold, then the face will be regarded as invisible

    Returns:

    """
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    for d in bpy.context.preferences.addons['cycles'].preferences.devices:
        if d["name"] == 'AMD Ryzen Threadripper PRO 3955WX 16-Cores':
            d["use"] = 0
        else:
            d["use"] = 1
        print(d["name"],d["use"])
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False, confirm=False)
    bpy.ops.import_scene.obj(filepath=model_path)
    bpy.ops.object.select_all(action='DESELECT')
    
    #calculate the areas of faces
    areas=[]
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for face in obj.data.polygons:
                areas.append(face.area)
    areas=np.array(areas)
    vis_mat=np.zeros((len(areas),view_points.shape[0]))
    #calculate the visibility of all the faces for each viewpoint
    vp_idx=0
    for point in view_points:
        direction=directions[vp_idx]
        vis_vect=getPointCoverage(consider_self_hit,point,direction,FOV,FOV_limit,vp_idx,dist_thres,angle_thres,vis_path,ground_faces)
        print('vis file saved '+str(vp_idx)+'/'+str(view_points.shape[0]),end='\r')
        #a=vis_mat[:][vp_idx]
        vis_mat[:,vp_idx]=vis_vect
        vp_idx+=1
    
    vis_times=np.sum(vis_mat,axis=1)
    vis_bool=np.where(vis_times==0,0,1)

    vis_area=np.dot(vis_bool,areas)
    return vis_area

def getPointCoverage(consider_self_hit,point,rotation,FOV,FOV_limit,vp_idx,dist_thres,angle_thres,vis_path,ground_faces):
    """
    Calculate the coverage of a viewpoint on the model
    Parameters:
        point - NDArray(3,1)  The position of viewpoint
        rotation - NDArray(3,1)  The rotation (direction) of viewpoint in radius
        FOV - Float  Field of view of the camera
        FOV_limit - Bool  If FOV limit is applied
        vp_idx - Int  The index of the viewpoint in the viewpoints list
        max_index - Int  The number of faces of the model
        dist_thres - Float  The distance threshold for judging if a face is visible. If the distance between the face and the
        viewpoint is larger than the threshold, then the face will be regarded as invisible
        angle_thres - Float  The inclination angle threshold for judging if a face is visible. If the inclination angle
        between the normal of the face and the ray direction is larger than the threshold, then the face will be regarded as invisible
    
    Returns: 
        vis_vect - NDArray(number of faces, 1)  The visibility of all the faces from the point
        direction_vect - NDArray(number of faces, )
    """
    #for some reason the point need to be rotated along x axis for 90 degrees

    cam = bpy.data.cameras.new("Camera1")
    cam_obj1 = bpy.data.objects.new("Camera1",cam)


    bpy.context.scene.collection.objects.link(cam_obj1)
    viewpoint_counter=0
    index_list=[]           
    direction=[] 
    min_dist=1000
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for face in obj.data.polygons:
                ray_direction=(np.asarray(face.center)-np.array([point[0], point[1], point[2]]))
                ray_direction=ray_direction/np.linalg.norm(ray_direction)
                direction.append(ray_direction)
                face.select=False

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for face in obj.data.polygons:
                #judge if the distance between viewpoint and face is larger than the threshold
                dist=np.sqrt((point[0]-face.center[0])**2+(point[1]-face.center[1])**2+(point[2]-face.center[2])**2)
                is_in_dist=(dist<dist_thres)
                if dist<min_dist:
                    nearest_face=face
                    min_dist=dist
                
                #judge if the inclination angle is larger than the threshold
                ray_direction=np.asarray(face.center)-point
                ray_direction=ray_direction/np.linalg.norm(ray_direction)
                face_normal=np.asarray(face.normal)
                if np.linalg.norm(face_normal)==0:
                    is_in_angle = True
                else:
                    is_in_angle = abs(np.dot(ray_direction,face_normal))>np.cos(angle_thres)
                camera_direction=getCosinesFromEuler(cam_obj1.rotation_euler.x,cam_obj1.rotation_euler.y,cam_obj1.rotation_euler.z)
                if ~FOV_limit:

                    is_in_FOV=True
                else:
                    is_in_FOV = np.dot(camera_direction,ray_direction)>np.cos(FOV)
                if is_in_dist and is_in_angle and is_in_FOV:
                    if consider_self_hit:
                        hit, location, normal, index = obj.ray_cast(point,ray_direction)
                    else:
                        hit=True
                        index=face.index
                    segment=np.array([[point],[face.center]])
                    
                    hit_ground=segmentsIntersectTriangles(segment, ground_faces)
                    if hit and index==face.index and ~hit_ground:
                        face.select=True
                        #vis_mat[viewpoint_counter][face.index]=1
                        index_list.append(face.index)  
                     
    cc=0
    vis=[]

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for face in obj.data.polygons:
                #print(cc,face.select)
                cc+=1
                vis.append([int(cc),int(face.select)])

    np.savetxt(vis_path+'vis'+str(vp_idx)+'.txt',vis,fmt='%i')
    np.savetxt(vis_path+'direction'+str(vp_idx)+'.txt',direction)
    np.savetxt(vis_path+'face'+str(vp_idx)+'.txt',np.asarray(nearest_face.center))
 
    viewpoint_counter+=1

    vis_np=np.array(vis)
    return vis_np[:,1]

def getCosinesFromEuler(roll,pitch,yaw):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])

    rotMat = Rz_yaw @ Ry_pitch @ Rx_roll
    return rotMat @ np.array([0,0,-1])

def cal_conn(point1,point2,env_coord_map,lin_parameters):
    '''
    Check if there is obstacle between two points
    '''
    xmin,xmax,ymin,ymax,zmin,zmax,step=lin_parameters
    point1_avl=env_coord_map[int(cal_idx_from_coord(point1,lin_parameters))][0]
    point2_avl=env_coord_map[int(cal_idx_from_coord(point2,lin_parameters))][0]
    if point1_avl==0 or point2_avl==0:
        return 0
    delta=point2-point1
    maxdelta=max(abs(delta[0]),abs(delta[1]),abs(delta[2]),1/step)
    deltastep=delta/maxdelta*step
    for stepidx in range(int(maxdelta/step)):
        pathpoint=point1+stepidx*deltastep
        pathpoint[0],pathpoint[1],pathpoint[2]=round((pathpoint[0]-xmin)/step)*step+xmin,round((pathpoint[1]-ymin)/step)*step+ymin,round((pathpoint[2]-zmin)/step)*step+zmin,
        if env_coord_map[int(cal_idx_from_coord(pathpoint,lin_parameters))][0]==0:
            return 0
    return 1
def cal_idx_from_coord(point,lin_parameters):
    #input the coordinate of point, output its index
    x,y,z=point[0],point[1],point[2]
    xmin,xmax,ymin,ymax,zmin,zmax,step=lin_parameters    
    idx=(z-zmin)/step+(y-ymin)/step*round((zmax-zmin)/step)+(x-xmin)/step*round((zmax-zmin)/step)*round((ymax-ymin)/step) #round or up round?
    return int(idx)
def is_in_distance(point,faces,vertices,dist_thres):
    """
    Determine if a point is in a certain distance of a set of faces

    Parameters:
        point - NDArray (3,1)
        faces - NDArray (num_faces,3)   The values of faces array are the index of vertices. Each face has 3 vertices.
        vertices - NDArray (num_vertices,3) The values of vertices are the xyz coordinates. 
        dist_thres - float  The threshold distance

    Returns:
        Bool value. True if the point is within the dist_thres for at least one face. Otherwise False.

    """
    count_intersect=0
    for face in faces:
        vertice1=vertices[face[0]]
        vertice2=vertices[face[1]]
        vertice3=vertices[face[2]]
        triangle=np.array([vertice1,vertice2,vertice3])
        
        if point_in_triangle((point[0],point[1]), (vertice1[0],vertice1[1]), (vertice2[0],vertice2[1]), (vertice3[0],vertice3[1])):
            if project_point_onto_triangle_along_z(triangle, point)[2]>point[2]:
                count_intersect+=1
        if pointTriangleDistance(triangle,point)[0]<dist_thres:
            #print('is in distance?: True')
            return True
    #print('is in distance?: False')
    if count_intersect%2==0:
        return False
    else:
        return True

def Write3dcoord_toply(point_array,path):
    """
    Write a point array to ply file

    Parameters:
        point_array - NDArray(n,3)  Coordinates of the points
        path - String Save path
    
    Returns:
        dist - Float The distance between two points
    """
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(point_array)
    o3d.io.write_point_cloud(path,pcd)

def pointPointDistance(P1,P2):
    """
    Calculate the distance between 2 points in 3D space

    Parameters:
        x1,y1,z1 - Float  Coordinates of the first point in 3D space
        x2,y2,z2 - Float  Coordinates of the second point in 3D space
    
    Returns:
        dist - Float The distance between two points
    """

    dist=np.sqrt((P1[0]-P2[0])**2+(P1[1]-P2[1])**2+(P1[2]-P2[2])**2)
    return dist


#https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e
#!/usr/bin/env python
#
# Tests distance between point and triangle in 3D. Aligns and uses 2D technique.
#
# Was originally some code on mathworks
def pointTriangleDistance(TRI, P):
    # function [dist,PP0] = pointTriangleDistance(TRI,P)
    # calculate distance between a point and a triangle in 3D
    # SYNTAX
    #   dist = pointTriangleDistance(TRI,P)
    #   [dist,PP0] = pointTriangleDistance(TRI,P)
    #
    # DESCRIPTION
    #   Calculate the distance of a given point P from a triangle TRI.
    #   Point P is a row vector of the form 1x3. The triangle is a matrix
    #   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
    #   dist = pointTriangleDistance(TRI,P) returns the distance of the point P
    #   to the triangle TRI.
    #   [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
    #   closest point PP0 to P on the triangle TRI.
    #
    # Author: Gwolyn Fischer
    # Release: 1.0
    # Release date: 09/02/02
    # Release: 1.1 Fixed Bug because of normalization
    # Release: 1.2 Fixed Bug because of typo in region 5 20101013
    # Release: 1.3 Fixed Bug because of typo in region 2 20101014

    # Possible extention could be a version tailored not to return the distance
    # and additionally the closest point, but instead return only the closest
    # point. Could lead to a small speed gain.

    # Example:
    # %% The Problem
    # P0 = [0.5 -0.3 0.5]
    #
    # P1 = [0 -1 0]
    # P2 = [1  0 0]
    # P3 = [0  0 0]
    #
    # vertices = [P1; P2; P3]
    # faces = [1 2 3]
    #
    # %% The Engine
    # [dist,PP0] = pointTriangleDistance([P1;P2;P3],P0)
    #
    # %% Visualization
    # [x,y,z] = sphere(20)
    # x = dist*x+P0(1)
    # y = dist*y+P0(2)
    # z = dist*z+P0(3)
    #
    # figure
    # hold all
    # patch('Vertices',vertices,'Faces',faces,'FaceColor','r','FaceAlpha',0.8)
    # plot3(P0(1),P0(2),P0(3),'b*')
    # plot3(PP0(1),PP0(2),PP0(3),'*g')
    # surf(x,y,z,'FaceColor','b','FaceAlpha',0.3)
    # view(3)

    # The algorithm is based on
    # "David Eberly, 'Distance Between Point and Triangle in 3D',
    # Geometric Tools, LLC, (1999)"
    # http:\\www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    #
    #        ^t
    #  \     |
    #   \reg2|
    #    \   |
    #     \  |
    #      \ |
    #       \|
    #        *P2
    #        |\
    #        | \
    #  reg3  |  \ reg1
    #        |   \
    #        |reg0\
    #        |     \
    #        |      \ P1
    # -------*-------*------->s
    #        |P0      \
    #  reg4  | reg5    \ reg6
    # rewrite triangle in normal form
    B = TRI[0, :]
    E0 = TRI[1, :] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[2, :] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = dot(E0, E0)
    b = dot(E0, E1)
    c = dot(E1, E1)
    d = dot(E0, D)
    e = dot(E1, D)
    f = dot(D, D)

    #print "{0} {1} {2} ".format(B,E1,E0)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                #region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0

    dist = sqrt(sqrdistance)

    PP0 = B + s * E0 + t * E1
    return dist, PP0

def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def point_in_triangle(p, a, b, c):
    s1=sign(p, a, b)
    s2=sign(p, b, c)
    s3=sign(p, c, a)
    b1 =  s1 < 0.0
    b2 =  s2 < 0.0
    b3 =  s3 < 0.0
    if s1==0 or s2==0 or s3==0:
        return False
    else:
        return ((b1 == b2) and (b2 == b3))
    
def find_plane_equation(triangle_vertices):
    A, B, C = triangle_vertices
    AB = B - A
    AC = C - A
    normal = np.cross(AB, AC)
    d = -np.dot(normal, A)
    return np.concatenate((normal, [d]))

def project_point_onto_triangle_along_z(triangle_vertices, point):
    plane_equation = find_plane_equation(triangle_vertices)
    x, y, _ = point
    z = -(plane_equation[0] * x + plane_equation[1] * y + plane_equation[3]) / plane_equation[2]
    return np.array([x, y, z])