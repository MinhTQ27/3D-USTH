import pyvista as pv
import numpy as np
import fast_simplification
mesh = pv.read("D:/Pyvista/hihi.ply")

# mesh = fast_simplification.simplify_mesh(mesh, target_reduction=0.9)
min_curv = mesh.curvature(curv_type='minimum')
max_curv = mesh.curvature(curv_type='maximum')
min_abs = np.fabs(min_curv)
max_abs = np.fabs(max_curv)

# make a curv with the highest curvature of each principal curv at each point
curv = np.zeros(34712)
for i in range(34712):
    j = min_abs[i]
    k = max_abs[i]
    if j > k:
        curv[i] = j
    else:
        curv[i] = k

cmin = np.percentile(curv, 3) #3
cmax = np.percentile(curv, 90) #90

def FindNeighbors(visitedPoint, borderPoint, biggerDelta, smallerDelta):
    while(len(visitedPoint) != 0):
        lst=[]
        for i in visitedPoint.values():
            if i not in borderPoint.values():
                n = mesh.point_neighbors(i)
                borderPoint[f'{i}']=i
                for j in n:
                    if j not in borderPoint.values():
                        if (curv[j] >= biggerDelta and curv[j] <= smallerDelta):
                            lst.append(j)

        for k in lst:
            if k not in visitedPoint.values():
                visitedPoint[f'{k}']=k

        for m in borderPoint.values():
            if m in visitedPoint.values():
                del visitedPoint[f'{m}']

# To check whether all of neighbor (n) are < a value or not
def Check(point, value):
    neighbor = mesh.point_neighbors(point)
    for i in neighbor:
        if curv[i] < value:
            return 1
    return 0

# Find point that: curv[point] < curv[maxpoint] and point NOT IN borderPoint and neighbor(point) >= 1.5 . Recursively with n times
def SecondMaxPoint(maxpoint, borderPoint, curv, n): 
    secondMaxCurv = float('-inf')                   
    secondMaxPoint = -1
    maxCurv = curv[maxpoint]

    for point, curvature in enumerate(curv):
        if curvature < maxCurv and curvature > secondMaxCurv:
            if point not in borderPoint.values():
                if Check(point, 0) == 0:
                    secondMaxCurv = curvature
                    secondMaxPoint = point
    # n=n-1
    if n <= 0 or secondMaxPoint == -1:
        return secondMaxPoint
    return SecondMaxPoint(secondMaxPoint, borderPoint, curv, n-1)

# Find points that in range of [a, b] and store that points in lst
lst = []
def FindPoint(bigger_than, smaller_than, borderPoint, curv): 
    for point, curvature in enumerate(curv):
        if curvature <= smaller_than and curvature >= bigger_than:
            if point not in borderPoint.values():
                lst.append(point)

# # Find a maximum point --> 

borderPoint = {}

borderPoint0={} 
visitedPoint0={}
maximum = np.argmax(curv)
visitedPoint0[f'{14969}']=14969
FindNeighbors(visitedPoint0, borderPoint0, 3.6, curv[maximum]) #4.9

for i in borderPoint0.values():
    borderPoint[f'{i}']=i

FindPoint(2.9, 3, borderPoint, curv)

# point1 = SecondMaxPoint(maximum, borderPoint, curv, 17)
# borderPoint1={} 
# visitedPoint1={}
# maximum = np.argmax(curv)
# visitedPoint1[f'{point1}']=point1
# FindNeighbors(visitedPoint1, borderPoint1, 5, curv[maximum])

# for i in borderPoint1.values():
#     borderPoint[f'{i}']=i

# point2 = SecondMaxPoint(maximum, borderPoint, curv, 4)
# borderPoint2={} 
# visitedPoint2={}
# visitedPoint2[f'{point2}']=point2
# FindNeighbors(visitedPoint2, borderPoint2, 4.0, curv[maximum])

# for i in borderPoint2.values():
#     borderPoint[f'{i}']=i

# point3 = SecondMaxPoint(maximum, borderPoint, curv, 8)
# borderPoint3={} 
# visitedPoint3={}
# visitedPoint3[f'{point3}']=point3
# FindNeighbors(visitedPoint3, borderPoint3, 4, curv[maximum])

# for i in borderPoint3.values():
#     borderPoint[f'{i}']=i

# point4 = SecondMaxPoint(maximum, borderPoint, curv, 11)
# borderPoint4={} 
# visitedPoint4={}
# visitedPoint4[f'{point4}']=point4
# FindNeighbors(visitedPoint4, borderPoint4, 3.95, curv[maximum])

# for i in borderPoint4.values():
#     borderPoint[f'{i}']=i

# point5 = SecondMaxPoint(maximum, borderPoint, curv, 12)
# borderPoint5={} 
# visitedPoint5={}
# visitedPoint5[f'{point5}']=point5
# FindNeighbors(visitedPoint5, borderPoint5, 4, curv[maximum])

# for i in borderPoint5.values():
#     borderPoint[f'{i}']=i

# point_conn_0 = SecondMaxPoint(maximum, borderPoint, curv, 12)
# connect = {}
# visited_connect = {}
# visited_connect[f'{point_conn_0}'] = point_conn_0
# # Connect(point_conn_0, connect, borderPoint)
# FindNeighbors(visited_connect, connect, 0, 5)

p = pv.Plotter()
p.add_mesh(mesh, show_edges=True, scalars=curv, clim=[cmin, cmax])
for i in borderPoint0.values():
        p.add_mesh(mesh.points[i], color="red", point_size=10)
# for i in borderPoint1.values():
#         p.add_mesh(mesh.points[i], color="blue", point_size=10)
# for i in borderPoint3.values():
#         p.add_mesh(mesh.points[i], color="white", point_size=10)
# for i in borderPoint4.values():
#     p.add_mesh(mesh.points[i], color="pink", point_size=10)
# for i in borderPoint5.values():
#     p.add_mesh(mesh.points[i], color="orange", point_size=10)
# for i in connect.values():
#     p.add_mesh(mesh.points[i], color="purple", point_size=10)
for i in lst:
        p.add_mesh(mesh.points[i], color="blue", point_size=10)

p.show(cpos="yx")