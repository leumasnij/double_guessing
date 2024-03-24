import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix
from scipy.interpolate import UnivariateSpline

def closest_points(set1, set2):
    min_distance = float('inf')
    closest_indices = None

    for i, point1 in enumerate(set1):
        for j, point2 in enumerate(set2):
            distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
            if distance < min_distance:
                min_distance = distance
                closest_indices = (i, j)

    return closest_indices, min_distance
def ConstructMST(sk):
    xs, ys = np.where(sk == 255)
    points = np.stack([xs, ys], axis=-1)
    for i in points:
        if np.sum(sk[i[0]-1:i[0]+2, i[1]-1:i[1]+2]) == 255:
            sk[i[0],i[1]] = 0
    xs, ys = np.where(sk == 255)
    points = np.stack([xs, ys], axis=-1)
    distances = distance.cdist(points, points, 'cityblock')
    distances = np.where(distances > 4, 0, distances)
    matrix = minimum_spanning_tree(distances).toarray()
    all_paths = []
    count = 0
    while np.sum(matrix):
        
        span_tree = csr_matrix(matrix + matrix.T)
        start_p = 0
        if connected_components(span_tree)[0] > 1:
            x = np.bincount(connected_components(span_tree)[1]).argmax()
            start_p = np.where(connected_components(span_tree)[1] == x)[0][0]
        path = find_furthest_path(span_tree, start_p, sk, points)
        path = find_furthest_path(span_tree, path[-1], sk, points)    
        pts = []
        last_node = 0
        for node in path:
            pts.append(points[node])
            if last_node == 0:
                last_node = node
            else:
                matrix[node, last_node] = 0
                matrix[last_node, node] = 0
                last_node = node 
        pts = np.array(pts)
        
        if(pts.shape[0] < 20):
            break
        count+=1
        all_paths.append(pts)

    return all_paths


def find_furthest_path(csr_matrix, start_node, sk, pts):
    num_nodes = csr_matrix.shape[0]
    visited = [False] * num_nodes
    distance = [-1] * num_nodes
    parent = [-1] * num_nodes
    st = []
    st.append(start_node)
    visited[start_node] = True
    distance[start_node] = 0
    count = 0
    while st:
        current_node = st[-1]
        st.remove(current_node)
        visited[current_node] = True
        count+=1
        
        # Get neighbors of the current node
        for neighbor in csr_matrix[current_node].indices:
          if not visited[neighbor]:
            if neighbor not in st:
                st.append(neighbor)
                distance[neighbor] = distance[current_node] + 1
                parent[neighbor] = current_node
            else:
                if distance[current_node] + 1 > distance[neighbor]:
                    distance[neighbor] = distance[current_node] + 1
                    parent[neighbor] = current_node
    furthest_node = np.argmax(distance)
    path = []
    while furthest_node != -1:
        path.insert(0, furthest_node)
        furthest_node = parent[furthest_node]
    return path

def dot_detection(input_image):
    
    x_list, y_list = np.where(input_image)
    width = np.max(x_list) - np.min(x_list)
    height = np.max(y_list) - np.min(y_list)
    area = input_image[np.min(x_list):np.max(x_list)+1,np.min(y_list):np.max(y_list)+1]
    area = np.where(area>0, 1.0, 0)
    precent = np.sum(area)/(width*height)
    if(precent>0.75):
        return True
    return False

def path_fusion(span):
    node = []
    edge = []
    for i in span:
        if node==[]:
            node.append([i[0], i[-1]])
            edge.append(i.copy())
        else:
            for j in range(len(node)):
                if math.sqrt(sum((xx - yy)** 2.0 for xx, yy in zip(node[j][0], i[0]))) < 8:
                    node[j][0] = i[-1]
                    edge[j] = np.append(i[::-1], edge[j], axis = 0)
                    break
                elif math.sqrt(sum((xx - yy)** 2.0 for xx, yy in zip(node[j][0], i[-1]))) < 8:
                    node[j][0] = i[0]
                    edge[j] = np.append(i, edge[j], axis = 0)
                    break
                elif math.sqrt(sum((xx - yy)** 2.0 for xx, yy in zip(node[j][-1], i[0]))) < 8:
                    node[j][-1] = i[-1]
                    edge[j] = np.append(edge[j], i, axis = 0)
                    break
                elif math.sqrt(sum((xx - yy)** 2.0 for xx, yy in zip(node[j][-1], i[-1]))) < 8:
                    node[j][-1] = i[0]
                    edge[j] = np.append(edge[j], i[::-1], axis = 0)
                    break
                elif j == len(node) - 1:
                    node.append([i[0], i[-1]])
                    edge.append(i.copy())
    return node, edge

def smoothen_and_to_list(span):
    node, edge = path_fusion(span)
    xy_list = []
    for i in edge:
        
        if(len(i) > 26):
            j = i.copy()
            flag = True
            while(len(j) > 26):
                i = j[:25].copy()
                j = j[25:]
                distances = [0.]
                for idx in range(len(i) - 1):
                    distances.append(distances[-1] + np.linalg.norm(i[idx] - i[idx+1]))
                distances = np.array(distances) / distances[-1]
                splines = [UnivariateSpline(distances, coords, k=5, s=len(i)) for coords in i.T]
                alpha = np.linspace(0., 1., len(i))
                smooth_pts = np.array([spl(alpha) for spl in splines]).T
                
                if flag:
                    xs = smooth_pts[:,0]
                    ys = smooth_pts[:,1]
                    flag = False
                else:
                    xs = np.append(xs, smooth_pts[:,0])
                    ys = np.append(ys, smooth_pts[:,1])
            distances = [0.]
            for idx in range(len(j) - 1):
                distances.append(distances[-1] + np.linalg.norm(j[idx] - j[idx+1]))
            distances = np.array(distances) / distances[-1]
            splines = [UnivariateSpline(distances, coords, k=min(len(j)-1, 5), s=len(j)) for coords in j.T]
            alpha = np.linspace(0., 1., len(j))
            smooth_pts = np.array([spl(alpha) for spl in splines]).T
            xs = np.append(xs, smooth_pts[:,0])
            ys = np.append(ys, smooth_pts[:,1])
            xy_list.append([xs, ys])
            continue
          

        
        distances = [0.]
        for idx in range(len(i) - 1):
            distances.append(distances[-1] + np.linalg.norm(i[idx] - i[idx+1]))
        distances = np.array(distances) / distances[-1]
        splines = [UnivariateSpline(distances, coords, k=5, s=len(i)) for coords in i.T]
        alpha = np.linspace(0., 1., len(i))
        smooth_pts = np.array([spl(alpha) for spl in splines]).T
        xs = smooth_pts[:,0]
        ys = smooth_pts[:,1]
        xy_list.append([xs, ys])
    return xy_list

def fill_part(mask):
    ret_list = []
    _, contour,hier = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask,[cnt],0,255,-1)
    cv2.imwrite('sample.jpg', mask.astype('uint8'))

    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    k_size = 60
    kernel = np.ones((k_size,k_size),np.uint8)
    while np.sum(erosion) > 255*400:
        components_list = []
        ret, labels = cv2.connectedComponents(erosion)
        for label in range(1,ret):
            im2 = np.zeros(labels.shape, dtype=np.uint8)
            if(len(np.where(labels == label)[0]) < 50):
                continue
            im2[labels == label] = 255
            components_list.append(im2.astype('uint8'))
        for i in range(len(components_list)):
            if dot_detection(components_list[i]) and np.sum(components_list[i])/255 < 100:
                x_list, y_list = np.where(components_list[i])
                center = ((np.min(x_list)+np.max(x_list))/2.0, (np.min(y_list)+np.max(y_list))/2.0)
                thickness = ((np.max(x_list)-np.min(x_list))+(np.max(y_list)-np.min(y_list)))/2
                ret_list.append([[np.array([center[0]]), np.array([center[1]])] , thickness])
                continue
            edges = cv2.Canny(components_list[i],100,200)
            span = ConstructMST(edges)
            xy_list = smoothen_and_to_list(span)
            ret_list.append(xy_list)
        kernel = np.ones((int(k_size*0.5),int(k_size*0.5)),np.uint8)
        erosion = cv2.erode(erosion,kernel,iterations = 1)
    ret_listx = ret_list[0][0][0]
    ret_listy = ret_list[0][0][1]
    #print(ret_listx)
    for i in range(len(ret_list)):
        if i+1==len(ret_list):
            break
        pointset1 = ret_list[i][0]
        pointset2 = ret_list[i+1][0]
        closest_pair, closest_distance = closest_points([np.array(pointset1).T[0]], np.array(pointset2).T)
        #print(pointset2)
        ret_list[i+1][0] = np.array([np.append(pointset2[0][closest_pair[1]:],(pointset2[0][:closest_pair[1]])), np.append(pointset2[1][closest_pair[1]:],(pointset2[1][:closest_pair[1]]))])
        pointset2 = ret_list[i+1][0]
        inter_x = []
        inter_y = []
        #print(closest_distance)
        for i in range(int(closest_distance)-1):
            if i == 0:
                inter_x.append(pointset1[0][-1] + (pointset2[0][0] - pointset1[0][-1])/closest_distance)
                inter_y.append(pointset1[1][-1] + (pointset2[1][0] - pointset1[1][-1])/closest_distance)
            else:

                inter_x.append(inter_x[-1] + (pointset2[0][0] - pointset1[0][-1])/closest_distance)
                inter_y.append(inter_y[-1] + (pointset2[1][0] - pointset1[1][-1])/closest_distance)

        
        ret_listx = np.append(ret_listx, np.array(inter_x))
        ret_listy = np.append(ret_listy, np.array(inter_y))
        ret_listx = np.append(ret_listx, pointset2[0])
        ret_listy = np.append(ret_listy, pointset2[1])
        
        #print(ret_listx)


    return [[ret_listx[::-1], ret_listy[::-1]]]

def outline(mask):
    ret, labels = cv2.connectedComponents(mask)
    components_list = []
    ret_list = []
    for label in range(1,ret):
        im2 = np.zeros(labels.shape, dtype=np.uint8)
        if(len(np.where(labels == label)[0]) < 50):
            continue
        im2[labels == label] = 255
        components_list.append(im2)
    for i in range(len(components_list)):

        if dot_detection(components_list[i]):
            x_list, y_list = np.where(components_list[i])
            center = ((np.min(x_list)+np.max(x_list))/2.0, (np.min(y_list)+np.max(y_list))/2.0)
            thickness = ((np.max(x_list)-np.min(x_list))+(np.max(y_list)-np.min(y_list)))/2
            ret_list.append([[np.array([center[0]]), np.array([center[1]])] , thickness])
            continue
        skeleton = cv2.ximgproc.thinning(components_list[i])
        skeleton = np.where(skeleton>10, 255, 0).astype('uint8')
        cv2.imwrite('sample_img_' + str(i)+ '.jpg', skeleton.astype('uint8'))
        thickness = np.sum(components_list[i].copy()*1.0)/np.sum(skeleton.copy())*1.0
        span = ConstructMST(skeleton)
        xy_list = smoothen_and_to_list(span)
        for j in xy_list:
            ret_list.append(j)
    return ret_list

def image_decomp_main(adr, mode ):
    img = cv2.imread(adr)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif  len(img.shape) != 1:
        print('unsupport image type')
        return
    img = cv2.resize(img, (250,250))
    if img[0][0] == 0:
        img = np.where(img==0, 255, img)
    background = np.average(img)
    mask = np.where(img<background, 255, 0).astype('uint8')
    if mode == 'outline':
        ret_list = outline(mask)
    elif mode == 'fill':
        ret_list = fill_part(mask)  
    return ret_list

        
if __name__ == '__main__':
    adrr = '/home/rocky/samueljin/pancake_bot/Images/file3.png'
    ret_list = image_decomp_main(adrr, 'fill')
    #print(ret_list[0])
    sample_figure = np.zeros((250,250))
    for i in range(len(ret_list)):
    #print(ret_list)
        x = ret_list[i][0]
        y = ret_list[i][1]
        for j in range(len(x)):
            sample_figure[int(x[j]), int(y[j])] = 255
        
    cv2.imwrite('sample_img.jpg', sample_figure.astype('uint8'))

    