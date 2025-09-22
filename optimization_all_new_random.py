import random
from deap import base, creator, tools
from utils import cal_idx_from_coord
import numpy as np
import open3d as o3d
from utils import  greedy_minimize_rows, calculate_coverage,Write3dcoord_toply, Write3dcoord_topath, output_visualizable_faces, waypoint_to_center, random_rotate, find_pairs
import os
import torch
import multiprocessing
import time

#define inputs
#import the viewpoint space
max_coverage=1.0
vis_dist=10
data_path='safedist=0.5_visdist='+str(vis_dist)+'_step=2/'
#data_path='dist='+str(vis_dist)+'/'
vp_path='pointcloud_out.ply'
pcd=o3d.io.read_point_cloud(data_path+vp_path)
view_points=np.asarray(pcd.points)
#boundary parameters
xmin,xmax,ymin,ymax,zmin,zmax,step=-23,0,-13,8,1,11,2
lin_parameters=(xmin,xmax,ymin,ymax,zmin,zmax,step)
START_POINT=np.array([xmin,ymin,zmin])

KAPPA=0.1  

#import neighbor matrix, viewpoint information, visibility matrix, direction matrix, area matrix
NEIGHBOR_MATRIX=np.loadtxt(data_path+'neighbor_matrix',dtype=int)
ENV_COORD_MAP=np.loadtxt(data_path+'env_mat')
CONN=np.loadtxt(data_path+'connection')
VIS_MAT=np.loadtxt(data_path+'filtered_vis_matrix')
DIR_MAT=np.load(data_path+'filtered_dir_matrix.npy')
AREA_ARR=np.loadtxt(data_path+'filtered_area_matrix')#.reshape((VIS_MAT.shape[0],1))
FACE_ARR=np.loadtxt(data_path+'face_matrix')
VIS_MAT_tensor=torch.tensor(VIS_MAT, dtype=torch.float32)
DIR_MAT_tensor=torch.tensor(DIR_MAT, dtype=torch.float32)
AREA_ARR_tensor=torch.tensor(AREA_ARR, dtype=torch.float32)
VIS_MAT_tensor = VIS_MAT_tensor.to('cuda')
DIR_MAT_tensor = DIR_MAT_tensor.to('cuda')
AREA_ARR_tensor = AREA_ARR_tensor.to('cuda')


def ga_opt(rule_proportion, population_size, exe_rate, new_proportion, happen_rate, tournament_rate, coverage_thres,save_path):
    isExist = os.path.exists(save_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_path)
    #define optimization parameters
    MUTATE_CHANGE_RATE=exe_rate
    MUTATE_ADD_RATE=exe_rate
    MUTATE_DELETE_RATE=exe_rate
    MUTATE_ROTATE_RATE=exe_rate
    MATE_RATE=exe_rate
    NEW_PROPORTION=new_proportion
    MUTATE_HAPPEN_RATE=happen_rate
    MATE_HAPPEN_RATE=happen_rate
    STRAIGHT_RATE=0.0
    (MIN_LENGTH, MAX_LENGTH)=200,600

    POPULATION_SIZE=population_size
    TOURNAMENT_SIZE = int(tournament_rate*POPULATION_SIZE)
    NUM_GENERATIONS=500

    #optimization settings include FOV_THRES, coverage_limit, 
    FOV_THRES=np.pi/3
    FOV_COS_THRES=np.cos(FOV_THRES)
    COVERAGE_LIMIT=coverage_thres
    
    z_deck=5
    x_ymin=[-18,-10]
    x_ymax=[-16, -7]
    boundary_parameters1=(xmin,x_ymin[0],xmin,x_ymax[0],z_deck)
    boundary_parameters2=(x_ymin[0]+step,x_ymin[1],x_ymax[0]+step,x_ymax[1],z_deck)
    boundary_parameters3=(x_ymin[1]+step,xmax-step,x_ymax[1]+step,xmax-step,z_deck)
    boundary_parameters=(boundary_parameters1,boundary_parameters2,boundary_parameters3)

    # Define the rule-based initialization function
    def initialization(save_path,boundary_parameters,num_paths,lin_parameters):
        def path_generater(waypoints,lin_parameters):
            """
            Generate a path(all the grid cells the path passed) with waypoints
            Parameters:
                waypoints - NDArray(n_waypoints,3)  the xyz coordinates of selected waypoints
                conn - NDArray(n_viewpoints,n_viewpoints) The connection information between all the viewpoints
                coord_map - NDArray(n_viewpoints,3)  The coordinate information of all the viewpoints
                step - Float  The grid size
            Returns:
                NDArray(n_path_viewpoints,3) The coordinates of all the viewpoints on the path

            """
            xmin,xmax,ymin,ymax,zmin,zmax,step=lin_parameters
            pathpoints=[]
            for wp_idx in range(waypoints.shape[0]-1):
                point1=waypoints[wp_idx]
                point2=waypoints[wp_idx+1]
                delta=point2-point1
                maxdelta=max(abs(delta[0]),abs(delta[1]),abs(delta[2]),1/step)
                deltastep=delta/maxdelta*step
                for stepidx in range(int(maxdelta/step)):
                    pathpoint=point1+stepidx*deltastep
                    pathpoint[0],pathpoint[1],pathpoint[2]=round((pathpoint[0]-xmin)/step)*step+xmin,round((pathpoint[1]-ymin)/step)*step+ymin,round((pathpoint[2]-zmin)/step)*step+zmin
                    pathpoints.append(pathpoint)
            pathpoints.append(point2)
            return np.asarray(pathpoints)
        def ran_pick_one_point(xmin,xmax,y,zmin,zmax,last_point,lin_parameters):

            if last_point is None:
                connected_indices = ENV_COORD_MAP[:, 0] == 1
                connected_points = ENV_COORD_MAP[connected_indices,1:4]
            else:
                last_idx=cal_idx_from_coord(last_point,lin_parameters)
                conn_last_point=CONN[last_idx]
                connected_indices= conn_last_point==1
                connected_points=ENV_COORD_MAP[connected_indices,1:4]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

            x_mask=np.logical_and(connected_points[:,0]>=xmin, connected_points[:,0]<=xmax)
            y_mask=connected_points[:,1]==y
            z_mask=np.logical_and(connected_points[:,2]>=zmin, connected_points[:,2]<=zmax)
            final_mask = np.logical_and(np.logical_and(x_mask, y_mask), z_mask)
            candidate_waypoints=connected_points[final_mask]
            if candidate_waypoints.shape[0]!=0:
                random_index = np.random.choice(candidate_waypoints.shape[0])
                random_waypoint=candidate_waypoints[random_index]
                return random_waypoint
            else:
                return None

        def pick_4_points(boundary_parameters,last_point,lin_parameters):
            xmin_ymin,xmax_ymin,xmin_ymax,xmax_ymax,z_deck=boundary_parameters
            point1,point2,point3,point4=None, None, None, None
            #pick waypoint for xmin-xmax, ymin, z under bridge.
            while point2 is None:
                point1=ran_pick_one_point(xmin_ymin,xmax_ymin,ymin,zmin,z_deck,last_point,lin_parameters)
                #move to ymax.
                point2=ran_pick_one_point(xmin_ymax,xmax_ymax,ymax-step,zmin,z_deck,point1,lin_parameters)
                    #move to z up
            while point4 is None:
                point3=ran_pick_one_point(xmin_ymax,xmax_ymax,ymax-step,z_deck,zmax-step,point2,lin_parameters)
                #move to ymin
                point4=ran_pick_one_point(xmin_ymin,xmax_ymin,ymin,z_deck,zmax-step,point3,lin_parameters)

            return np.array([point1,point2,point3,point4,point1])
        def rule_based_init_path(boundary_parameters,lin_parameters):
            boundary_parameters1,boundary_parameters2,boundary_parameters3=boundary_parameters
            points1=pick_4_points(boundary_parameters1,None,lin_parameters)
            points2=pick_4_points(boundary_parameters1,points1[-1],lin_parameters)
            points3=pick_4_points(boundary_parameters2,points2[-1],lin_parameters)
            points4=pick_4_points(boundary_parameters2,points3[-1],lin_parameters)
            points5=pick_4_points(boundary_parameters3,points4[-1],lin_parameters)
            points6=pick_4_points(boundary_parameters3,points4[-1],lin_parameters)
            waypoints = np.concatenate((points1, points2, points3,points4,points5,points6), axis=0)
            path=path_generater(waypoints,lin_parameters)

            new_individual=[path]

            return creator.Individual(new_individual)

        xmin,xmax,ymin,ymax,zmin,zmax,step=lin_parameters
        path_list=[]

        for count in range(num_paths):
            path_list.append(rule_based_init_path(boundary_parameters,lin_parameters))
        

        return path_list

    # Define the individual creation function
    def create_individual():
        
        (xmin,xmax,ymin,ymax,zmin,zmax,step)=lin_parameters
        START_POINT=view_points[np.random.randint(view_points.shape[0])]
        num_waypoints=random.randint(MIN_LENGTH, MAX_LENGTH)
        path=np.zeros((num_waypoints,3))
        waypoint_idx_list=[]
        last_point=START_POINT
        moving_dir=np.array([random.choice([-1,0,1]),random.choice([-1, 0, 1]),random.choice([-1, 0, 1])])*step
        for idx_waypoint in range(num_waypoints):
            last_idx=cal_idx_from_coord(last_point,lin_parameters)
            neighbors=NEIGHBOR_MATRIX[last_idx,:]
            neighbors=neighbors[neighbors!=-1]
            for num in neighbors:
                if num in waypoint_idx_list and num!=last_idx:
                    neighbors = np.delete(neighbors, np.where(neighbors == num))
            straight_point=last_point+moving_dir
            is_in_range=(xmin<=straight_point[0]<xmax) and (ymin<=straight_point[1]<ymax) and (zmin<=straight_point[2]<zmax)
            if is_in_range:
                straight_idx=cal_idx_from_coord(straight_point,lin_parameters)                                
            if is_in_range and straight_idx in neighbors and random.random()<STRAIGHT_RATE:
                current_idx=straight_idx
            else:   
                current_idx=np.random.choice(neighbors)
            waypoint_idx_list.append(current_idx)
            current_point=ENV_COORD_MAP[current_idx,1:4]
            path[idx_waypoint,:]=current_point
            moving_dir=current_point-last_point
            last_point=current_point
        new_individual=[path]
        #print('Create time:', time.time()-start_time)
        return creator.Individual(new_individual)
    # Define the mutation function for changing the coordinate of a point
    def mutate_change(individual):
        path=np.asarray(individual[0])        
        last_point=path[0]
        new_path=np.zeros((0,3))
        new_path_idx=np.array([]) 
        #change the coordinate of a point
        for waypoint_idx in range(path.shape[0]-1):
            if waypoint_idx!=0:
                last_point=new_path[waypoint_idx-1]
            current_point=path[waypoint_idx,:]
            next_point=path[waypoint_idx+1,:]
            current_idx=cal_idx_from_coord(current_point,lin_parameters)
            last_idx=cal_idx_from_coord(last_point,lin_parameters)
            next_idx=cal_idx_from_coord(next_point,lin_parameters)

            last_neighbors=NEIGHBOR_MATRIX[last_idx,:]
            next_neighbors=NEIGHBOR_MATRIX[next_idx,:]
            
            #change the coordinate of a point
            if random.random()<MUTATE_CHANGE_RATE:
                #find all the common neighbors of last_point and next_point
                last_next_neighbors=np.array(np.intersect1d(last_neighbors, next_neighbors))
                #delete the unavailable points
                last_next_neighbors = np.delete(last_next_neighbors, np.where(last_next_neighbors == -1))
                #delete the points that already exist to get rid of returning to the same point
                #last_next_neighbors = np.setdiff1d(last_next_neighbors, new_path)

                #if there is any common neighbor, choose one randomly. Otherwise keep the original one
                if last_next_neighbors.shape[0]!=0:
                    idx=np.random.choice(last_next_neighbors)
                    new_point=ENV_COORD_MAP[idx,1:4]
                    new_path=np.append(new_path,[new_point],axis=0)
                    new_path_idx=np.append(new_path_idx,idx)
                else:                
                    new_path=np.append(new_path,[current_point],axis=0)
                    new_path_idx=np.append(new_path_idx,current_idx)

            else:
                new_path=np.append(new_path,[current_point],axis=0)
                new_path_idx=np.append(new_path_idx,current_idx)
        #print('Change time:', time.time()-start_time)
        return creator.Individual([new_path])
    # Define the mutation function for adding a point
    def mutate_add(individual):
        start_time=time.time()
        path=np.asarray(individual[0])

        last_point=path[0]
        new_path=np.zeros((0,3))

        for waypoint_idx in range(path.shape[0]):
            if waypoint_idx!=0:
                last_point=new_path[-1]        
            current_point=path[waypoint_idx,:]
            current_idx=cal_idx_from_coord(current_point,lin_parameters)
            last_idx=cal_idx_from_coord(last_point,lin_parameters)

            current_neighbors=NEIGHBOR_MATRIX[current_idx,:]
            last_neighbors=NEIGHBOR_MATRIX[last_idx,:]

            #add a new point before current point             
            if random.random()<MUTATE_ADD_RATE:
                #find all the common neighbors of last_point and current_point
                last_current_neighbors=np.array(np.intersect1d(last_neighbors, current_neighbors))
                last_current_neighbors = np.delete(last_current_neighbors, np.where(last_current_neighbors == -1))
                #if there is any common neighbor, choose one to add randomly.
                if last_current_neighbors.shape[0]!=0:
                    idx=np.random.choice(last_current_neighbors)
                    new_point=ENV_COORD_MAP[idx,1:4]
                    new_path = np.append(new_path, [new_point], axis=0)

            #no matter add a new point or not, add the original current point
            new_path = np.append(new_path, [current_point], axis=0)
        #print('Add time:', time.time()-start_time)
        return creator.Individual([new_path])
    
    # Define the mutation function for deleting a point
    def mutate_delete(individual):
        start_time=time.time()
        path=np.asarray(individual[0])
        last_point=path[0]
        new_path=np.zeros((0,3))
        for waypoint_idx in range(path.shape[0]-1):
            if waypoint_idx!=0 and new_path.shape[0]!=0:
                last_point=new_path[-1]        
            current_point=path[waypoint_idx,:]
            next_point=path[waypoint_idx+1,:]
            current_idx=cal_idx_from_coord(current_point,lin_parameters)
            last_idx=cal_idx_from_coord(last_point,lin_parameters)
            next_idx=cal_idx_from_coord(next_point,lin_parameters)
            current_neighbors=NEIGHBOR_MATRIX[current_idx,:]
            last_neighbors=NEIGHBOR_MATRIX[last_idx,:]
            next_neighbors=NEIGHBOR_MATRIX[next_idx,:]  
            #if last_point and next_point are neighbors, give a probability to delete the current_point          
            if random.random()<MUTATE_DELETE_RATE and last_idx in next_neighbors:
                pass
            else:
                new_path = np.append(new_path, [current_point], axis=0)            
        #print('Delete time:', time.time()-start_time)
        return creator.Individual([new_path]) 
    
    #rotate won't be applied in this version
    def mutate_rotate(individual):
        start_time=time.time()
        path=np.asarray(individual[0])
        sight_directions=np.asarray(individual[1])

        for idx in range(sight_directions.shape[0]):
            if random.random()<MUTATE_ROTATE_RATE:
                current_idx=cal_idx_from_coord(path[idx],lin_parameters)
                face=FACE_ARR[current_idx]
                sight_directions[idx,:]=waypoint_to_center(path[idx],face)
                new_sight=random_rotate(sight_directions[idx,:],KAPPA)
                norm = np.linalg.norm(new_sight)
                new_sight=new_sight/norm
                sight_directions[idx,:]=new_sight
                

        #print('Rotate time:', time.time()-start_time)
        return creator.Individual([path,sight_directions])

    #check if a path is consecutive
    def check_connection(individual):  
        connection=True
        path=np.asarray(individual)
        last_point=START_POINT
        for waypoint_idx in range(path.shape[0]-1):
            current_point=path[waypoint_idx,:]
            delta=current_point-last_point
            if abs(delta[0])>step or abs(delta[1])>step or abs(delta[2])>step:
                connection=False
            last_point=current_point
        return connection
    #check if there is loop in a path. Will do this in the future

    # Define the mate function
    def mate(individual1,individual2):
        start_time=time.time()
        path1=np.asarray(individual1[0])
        path2=np.asarray(individual2[0])

        path1_idx=[]
        path2_idx=[]
        for point1 in path1:
            path1_idx.append(cal_idx_from_coord(point1,lin_parameters))
        for point2 in path2:
            path2_idx.append(cal_idx_from_coord(point2,lin_parameters))
        #print('Calculate idx time:', time.time()-start_time)
        
        #find all the intersection points between two paths
        neighbors1=NEIGHBOR_MATRIX[path1_idx]
        #print('Neighbor time:', time.time()-start_time)
        pairs_local=find_pairs(neighbors1,path2_idx)
        '''
        for point1_idx in path1_idx:
            neighbors=NEIGHBOR_MATRIX[point1_idx,:]
            point2_local_idx=0
            for point2_idx in path2_idx:
                if point2_idx in neighbors:
                    pairs_local.append([point1_local_idx,point2_local_idx])
                point2_local_idx+=1
            point1_local_idx+=1
        '''
        #print('Find pairs time:', time.time()-start_time)
        if pairs_local.shape[0]==0:
            #print('no swap')
            return creator.Individual([path1])
        else:
            pairs_local_1=pairs_local
            pairs_local_2=pairs_local_1[pairs_local_1[:, 1].argsort(kind='mergesort')][:, ::-1][pairs_local_1[:, 0].argsort(kind='mergesort')][:, ::-1]
            current_path=1
            path1_pointer=0
            path2_pointer=0
            selected_pairs_local=[]
            while path1_pointer!=[pairs_local_1[-1,0]] and path2_pointer!=[pairs_local_2[-1,1]]:
                if current_path==1:
                    path1_pointer = pairs_local_1[np.argmax(pairs_local_1[:, 0] > path1_pointer)][0]
                    if random.random()<MATE_RATE:
                        candidate_pairs_local = pairs_local_1[pairs_local_1[:, 0] == path1_pointer]
                        candidate_pairs_local = candidate_pairs_local[candidate_pairs_local[:, 1] > path2_pointer]
                        if candidate_pairs_local.shape[0]:
                            selected_pair_local = candidate_pairs_local[np.random.choice(candidate_pairs_local.shape[0])]
                            selected_pairs_local.append(selected_pair_local)
                            path2_pointer=selected_pair_local[1]
                            current_path=2
                else:
                    path2_pointer = pairs_local_2[np.argmax(pairs_local_2[:, 1] > path2_pointer)][1]
                    if random.random()<MATE_RATE:
                        candidate_pairs_local = pairs_local_2[pairs_local_2[:, 1] == path2_pointer]
                        candidate_pairs_local = candidate_pairs_local[candidate_pairs_local[:, 0] > path1_pointer]
                        if candidate_pairs_local.shape[0]:
                            selected_pair_local = candidate_pairs_local[np.random.choice(candidate_pairs_local.shape[0])]
                            selected_pairs_local.append(selected_pair_local)
                            path1_pointer=selected_pair_local[0]
                            current_path=1
            #print(selected_pairs_local)
            current_path=1
            pointer_1=0
            pointer_2=0
            new_path=np.zeros((0,3))

            for pair in selected_pairs_local:
                if current_path==1:            
                    new_path = np.concatenate((new_path,path1[pointer_1:pair[0]+1,:] ), axis=0)
                    current_path=2
                else:
                    new_path = np.concatenate((new_path,path2[pointer_2:pair[1]+1,:] ), axis=0)
                    current_path=1
                pointer_1,pointer_2=pair
            #add the last part
            if current_path==1:
                new_path = np.concatenate((new_path,path1[pointer_1:,:] ), axis=0)
            else:
                new_path = np.concatenate((new_path,path2[pointer_2:,:] ), axis=0)
            #print(new_path)
            #print('Swap time:', time.time()-start_time)

            return creator.Individual([new_path])
    # Define fitness function
    def fitness(individual):
        # Add the start and end positions to the path

        path_tensor = torch.tensor(individual[0], dtype=torch.float32)
        path_tensor = path_tensor.to('cuda')
        # Calculate the fitness of the path
        total_distance = 0
        coverage,vis_arr=calculate_coverage(path_tensor,None,VIS_MAT_tensor,DIR_MAT_tensor,lin_parameters,AREA_ARR_tensor,FOV_COS_THRES)
        coverage=coverage/max_coverage
        if coverage<COVERAGE_LIMIT:
            if coverage.item()==0:
                fitness=1000000000
            else:
                fitness=1000000/coverage.item()
        else: 
            movement_tensor = path_tensor[1:] - path_tensor[:-1]
            distance_tensor = torch.norm(movement_tensor, dim=1)
            fitness=torch.sum(distance_tensor).item()

        return fitness,

    #pool = multiprocessing.Pool()
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin,)
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", create_individual)
    
    toolbox.register("mutate_change", mutate_change)
    toolbox.register("mutate_add", mutate_add)
    toolbox.register("mutate_delete", mutate_delete)
    toolbox.register("mutate_rotate", mutate_rotate)
    toolbox.register("mate", mate)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
    toolbox.register("evaluate", fitness)
    #toolbox.register("map", pool.map)
    #rule_init=initialization(data_path,boundary_parameters,int(population_size*rule_proportion),lin_parameters)
    population = toolbox.population(n=int((1-rule_proportion)*population_size))
    population=population#+rule_init
    fitness_record=[]
    for generation in range(NUM_GENERATIONS):
        # Evaluate the fitness of individuals in the population
        offspring = []
        # Select parents for mating
        for offspring_idx in range(POPULATION_SIZE):
            if random.random()<NEW_PROPORTION:
                child=toolbox.individual()
                
            else:
                start_time=time.time()
                if random.random()<MATE_HAPPEN_RATE:
                    parent1, parent2 = toolbox.select(population, 2)
                    child = toolbox.mate(parent1, parent2)
        
                else:
                    child = toolbox.select(population,1)[0]
                if random.random()<MUTATE_HAPPEN_RATE:
                    child = toolbox.mutate_change(child)
                if random.random()<MUTATE_HAPPEN_RATE:
                    child = toolbox.mutate_add(child)
                if random.random()<MUTATE_HAPPEN_RATE:
                    child = toolbox.mutate_delete(child)
                #if random.random()<MUTATE_HAPPEN_RATE:
                    #child = toolbox.mutate_rotate(child)
        
            offspring.append(child)
            
            #print('Generation:', generation, 'Child:', offspring_idx)
        fits = toolbox.map(toolbox.evaluate, offspring)
        #fits=population_fitness(offspring)

        
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
            #print('Fitness time:', time.time()-start_time)
        population[:]=offspring
        best_individual = tools.selBest(population, k=1)[0]
        print("Generation: ", generation, " Fitness: ", best_individual.fitness.values[0])
        fitness_record.append(best_individual.fitness.values[0])
        if generation==0:
            global_best_individual=best_individual
        else:
            if global_best_individual.fitness.values[0]>best_individual.fitness.values[0]:
                global_best_individual=best_individual
    np.savetxt(save_path+'fitness.txt',np.asarray(fitness_record))
    global_best_individual_path=np.asarray(global_best_individual[0])
    global_best_individual_path_tensor=torch.tensor(global_best_individual[0], dtype=torch.float32)
    global_best_individual_path_tensor=global_best_individual_path_tensor.to('cuda')
    np.save(save_path+'test_path.npy', np.asarray(global_best_individual))
    Write3dcoord_toply( global_best_individual_path,save_path+'test_path.ply')
    Write3dcoord_topath( global_best_individual_path,save_path+"lines.ply")
    #coverage,vis_arr=calculate_coverage(path,sight,VIS_MAT,DIR_MAT,lin_parameters,AREA_ARR,FOV_THRES)
    coverage,vis_arr=calculate_coverage( global_best_individual_path_tensor,None,VIS_MAT_tensor,DIR_MAT_tensor,lin_parameters,AREA_ARR_tensor,FOV_COS_THRES)
    print(coverage,vis_arr)
    model_path='visualizable_cut.obj'
    output_visualizable_faces(model_path,vis_arr,save_path)

    #calculate the sights by greedy algorithm
    path=global_best_individual_path
    idx_list=[]
    for viewpoint in path:
        idx=cal_idx_from_coord(viewpoint, lin_parameters)
        idx_list.append(idx)
    #visibility of each face from each viewpoint
    viewpoints_vis=VIS_MAT[:,idx_list]
    #select all the rows for visible faces
    nonzero_rows = np.any(viewpoints_vis != 0, axis=1)
    viewpoints_vis_nonzero = viewpoints_vis[nonzero_rows]
    directions_viewpoints = DIR_MAT[idx_list,:,:]
    directions_viewpoints_nonzero=directions_viewpoints[:,nonzero_rows,:]
    viewpoints_vis_bool = viewpoints_vis_nonzero != 0
    vis_3d_new= np.empty((viewpoints_vis_bool.shape[0], viewpoints_vis_bool.shape[1], viewpoints_vis_bool.shape[0]),dtype=bool)

    for viewpoint_idx in range(directions_viewpoints.shape[0]):
        #vector1=directions_viewpoints[viewpoint_idx,:,:]
        vector2=directions_viewpoints_nonzero[viewpoint_idx,:,:]
        dot_products=np.dot(vector2,vector2.T)
        dot_bool=dot_products>FOV_COS_THRES
        dot_bool=dot_bool.T        
        vis_vector=viewpoints_vis_bool[:,viewpoint_idx]
        new_vis_slice=dot_bool & vis_vector
        vis_3d_new[:,viewpoint_idx,:]=new_vis_slice
    selected_row_indices_array,selected_matrix = greedy_minimize_rows(vis_3d_new)
    whole_path=[]
    indices = np.argsort(selected_row_indices_array[:, 1])
    sorted_points=selected_row_indices_array[indices]
    for point in sorted_points:
        face_idx=point[0]
        point_idx=point[1]
        position=path[point_idx]
        sight=directions_viewpoints_nonzero[point_idx,face_idx,:]
        whole_path.append([position,sight])
    np.save(save_path+'test_whole_path.npy', np.asarray(whole_path))
#rate_list=[0.1, 0.2, 0.4]
#proportion_list=[0.1, 0.2, 0.4]
#coverage_list=[0.8, 0.95]
exe_rate_list=[0.1]
new_proportion_list=[0]
happen_rate_list=[0.75]
tournament_rate_list=[0.1]
coverage_list=[0.95]
population_list=[250]
rule_list=[0.0]
for idx in range(0,1):
#for idx in range(1,2):
#for idx in range(2,3):
#for idx in range(3,4):
#for idx in range(4,5):
#for idx in range(5,6):
#for idx in range(6,7):
#for idx in range(7,8):
#for idx in range(8,9):
#for idx in range(9,10):
    for coverage in coverage_list:
        for tournament_rate in tournament_rate_list:
            for happen_rate in happen_rate_list:            
                for new_proportion in new_proportion_list:
                    for exe_rate in exe_rate_list:
                        for population in population_list:
                            for rule_proportion in rule_list:
                                ga_opt(rule_proportion, population, exe_rate, new_proportion, happen_rate, tournament_rate, coverage,'coarse/ruleinit='+str(rule_proportion)+'_KAPPA=0.1_FOV=120_dist=10_ind='+str(population)+'_gen=1000/'+str(exe_rate)+'_'+str(new_proportion)+'_'+str(happen_rate)+'_'+str(tournament_rate)+'_'+str(coverage)+'/'+str(idx+1)+'/')


'''
individual1=create_individual()
fitness=fitness(individual1)

for idx in range(100):
    individual1=create_individual()
    individual2=create_individual()
    new_ind=mate(individual1,individual2)
    is_connected=check_connection(new_ind)
    ind1=np.asarray(individual1)
    ind2=np.asarray(individual2)
    nind=np.asarray(new_ind)    
    print(ind1.shape[0],ind2.shape[0],nind.shape[0],is_connected)
    #is_connected_0=check_connection(individual)
    #new_ind=mutate_delete(individual)
    is_connected=check_connection(new_ind)
'''