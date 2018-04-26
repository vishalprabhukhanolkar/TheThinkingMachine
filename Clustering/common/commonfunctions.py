"""
Author: Vishal P
Common file used for all clustering notebooks
"""
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

#################################### visualize_cluster_data ####################################
# Visualize the data - common method used by all the clustering notebooks
def visualize_cluster_data(plt, data, cluster, title, centroids=[], filename = None):   
    
    # Set image parameters
    f, axs = plt.subplots(1,1,figsize=(8,6))
    axs.spines["top"].set_visible(False)  
    axs.spines["right"].set_visible(False)     
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)       
    
    # Plot the data points
    plt.scatter(x=data[:,0], y=data[:,1], c=cluster, cmap="brg", marker='o')    

    # Plot the centroids in black
    for cx, cy in centroids:
        plt.scatter(x=cx, y=cy, color='black' , marker='o')

    # Make the title bigger
    plt.title(title, fontsize=20)
    
    # Save the plot to file
    if (filename):
        f.savefig(filename + '.png')
        
    plt.show()
#################################### visualize_cluster_data ####################################


#################################### visualize_elbow_chart #####################################
# Visualization for elbow chart
def visualize_elbow_chart(plt, data, cluster_range, filename, random_state):
    inertia = [] # one value per cluster
    
    # Set image parameters
    f, axs = plt.subplots(1,1,figsize=(8,6))
    axs.spines["top"].set_visible(False)  
    axs.spines["right"].set_visible(False)     
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)

    # Calculate inertia values
    for i in cluster_range: # checking for 1 to 11 clusters
        kmeans = KMeans(n_clusters = i, random_state = random_state)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # Plot the elbow graph
    plt.plot(cluster_range, inertia)
    plt.title('Elbow chart',fontsize = 20)
    plt.xlabel('Number of clusters',fontsize = 16)
    plt.ylabel('Inertia',fontsize = 16)
    
    # Save the plot to file
    if (filename):
       f.savefig(filename + ".png") 
    
    plt.show()  
#################################### visualize_elbow_chart #####################################


#################################### visualize_knee_chart ######################################    
# Used by DBScan method
def visualize_knee_chart(plt, X, title, filename = None):
    neighbors = 5
    distances, indices = NearestNeighbors(n_neighbors=neighbors).fit(X).kneighbors(X)
    distanceDec = sorted(distances[:,neighbors-1])
    f, ax = plt.subplots(1,1,figsize=(8,6))
    ax.grid(color='gray', linestyle='-', linewidth=1)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14) 
    plt.plot(indices[:,0], distanceDec)
    plt.title(title, fontsize=20)
    if (filename):
        f.savefig(filename + '.png')
    plt.show()    
#################################### visualize_knee_chart ######################################   

################################  visualize_cluster_centroids  #################################    
# Visualize cluster data with centroids - Used in agglomerative example
def visualize_cluster_centroids(plt, data, cluster, title, centroids=False, filename = None):
  
    f, axs = plt.subplots(1,1,figsize=(8,6))
    axs.spines["top"].set_visible(False)  
    axs.spines["right"].set_visible(False)     
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
   
    plt.scatter(x=data[:,0], y=data[:,1], c=cluster, cmap="brg", marker='o')
    
    if (centroids):
        total_clusters = len(set(cluster))        
        
        for c in range(total_clusters):
            # Find the centroid of this cluster
            cx, cy, radius = find_centroid_maxdistance(data[cluster == c, 0], data[cluster == c, 1])            
            
            # Draw a circle around data points
            circle = plt.Circle((cx, cy), radius + 0.2 , color='black', fill = False)       
            axs.add_artist(circle)          
    
    plt.title(title, fontsize=20)
    
    if (filename):
        f.savefig(filename + '.png')
    
    plt.show()
################################  visualize_cluster_centroids  #################################       

######################################  visualize_dendrogram  ##################################    
# Visualization for dendrogram
def visualize_dendrogram(plt, data, filename=None):
    import scipy.cluster.hierarchy as sch
    
    f, axs = plt.subplots(1,1,figsize=(8,6))
    axs.spines["top"].set_visible(False)  
    axs.spines["right"].set_visible(False)     
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    
    dendrogram = sch.dendrogram(sch.linkage(data, method ='ward'))
    plt.title('Dendrogram', fontsize=20)
    
    if (filename):
        f.savefig(filename + '.png')
    
    plt.show()
######################################  visualize_dendrogram  ##################################        


############################  Helper methods for K-means  ######################################    
# Euclidean distance
def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2) * (x1-x2) + (y1-y2) * (y1-y2))

# Used to visualize the cluster circles
def find_max_cluster_distance(X, Y, cx, cy):
    distances = []
    for x, y in zip(X, Y):
        # Find dstnace from centroid
        distances.append(distance(x,y,cx,cy))
    
    # return index of the maximum distane
    return max(distances)
    
# Used to find centre of the cluster    
def find_centroid_maxdistance(X, Y):
    cx = sum(X)/len(X)
    cy = sum(Y)/len(Y) 
    
    return (cx,cy,find_max_cluster_distance(X, Y, cx, cy)) 
############################  Helper methods for K-means  ######################################    

############################  Helper methods for K-means exmaple  ##############################    
# used to find closest centroid for a given point
def find_centroid(x, y, centroids):
    distances = []
    for cx, cy in centroids:
        # Find dstnace from each centroid
        distances.append(distance(x,y,cx,cy))

    # return index of the minimum distane,
    return distances.index(min(distances))

# Find new centroid for points belonging to a cluster
def find_new_centroid(X, Y):
    cx = sum(X)/len(X)
    cy = sum(Y)/len(Y)
    return (cx,cy)
############################  Helper methods for K-means exmaple  ##############################      

############################  Helper methods for K-means exmaple  ##############################      
# Find the new cluster centre based on current cluster
def find_new_clusters(clusters, radius):
    new_clusters = []
    
    for x,y in clusters:        
        new_clusters.append(identify_new_location(x,y, clusters, radius))
        
    return new_clusters

# Find the new cluster centre based on density
def identify_new_location(x, y, clusters, radius):
    sum_nx = 0
    sum_ny = 0
    count = 0
    
    for dx, dy in clusters:                
        if (distance(x, y, dx, dy) <= radius):
            sum_nx = sum_nx + dx
            sum_ny = sum_ny + dy
            count = count + 1
    
    return [sum_nx/count, sum_ny/count]

# Visualize the data
def visualize_cluster_data_meanshift(plt, data, cluster, title, centroids=[], filename = None, showCircle = False, r = 0, c_index = 0):       
    # Define chart parameters
    f, axs = plt.subplots(1,1,figsize=(8,6))
    axs.spines["top"].set_visible(False)  
    axs.spines["right"].set_visible(False)     
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)       
    
    # show all data points
    plt.scatter(x=data[:,0], y=data[:,1], c=cluster, cmap="brg", marker='o')        
    
    # show all centroids 
    for cx, cy in centroids:
        plt.scatter(x=cx, y=cy, color='red' , marker='o')

    # show circle around one of the data points
    if (showCircle):
        cx,cy = centroids[c_index]
        plt.scatter(x=cx, y=cy, color='gray' , marker='o')
        circle = plt.Circle((cx, cy), radius= r, color='black', fill=False)
        axs.add_artist(circle)  
    
    # big title
    plt.title(title, fontsize=20)
    
    # save the image
    if (filename):
        f.savefig(filename + '.png')
        
    plt.show()  
############################  Helper methods for K-means exmaple  ##############################      


############################  run_visualization_example  #######################################    
# Loop over input data and perform k-means for different number of clusters and centriods
# This function is used by k-means example
def run_visualization_example (plt, number_of_clusters, data, centroids, filename):
    clusters = [0] * data.shape[0]
    
    visualize_cluster_data(plt, data, clusters, title = 'Raw data', centroids=centroids, filename=filename)
    
    # Repeat 10 times
    for outer in range (0, 10):
        # For all the data points calculate the closest centroid
        for index in range(0, data.shape[0]):
            clusters[index] = find_centroid(data[index,0], data[index,1], centroids)

        # Visualize the clusters
        visualize_cluster_data(plt, data, clusters, title='After points allocated to cluster', centroids=centroids, filename = filename +'_' + (str)(outer * 2 + 1).zfill(2) )

        # For each cluster identify the new centroid
        for cluster in range(0, number_of_clusters):
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster]
            if (cluster_indices):
                centroids[cluster] = find_new_centroid(data[cluster_indices,0], data[cluster_indices,1])

        # Visualize the clusters
        visualize_cluster_data(plt, data, clusters, title='After identifying new centers', centroids=centroids, filename = filename +'_' + (str)(outer * 2 + 2).zfill(2))
############################  run_visualization_example  #######################################    