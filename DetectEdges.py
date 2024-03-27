import pyvista as pv
import numpy as np
mesh = pv.read("D:/Pyvista/hihi.ply")
curv = mesh.curvature(curv_type='mean')
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

def Check(point, value):
    neighbor = mesh.point_neighbors(point)
    for i in neighbor:
        if curv[i] < value:
            return 1
    return 0

def SecondMaxPoint(maxpoint, borderPoint, curv, n): # Find a point that which curv is smaller than a chosen point and it's neighbor do not smaller than a given value
    secondMaxCurv = float('-inf')
    secondMaxPoint = -1
    maxCurv = curv[maxpoint]

    for point, curvature in enumerate(curv):
        if curvature < maxCurv and curvature > secondMaxCurv:
            if point not in borderPoint.values():
                if Check(point, 1.5) == 0:
                    secondMaxCurv = curvature
                    secondMaxPoint = point
    # n=n-1
    if n <= 0 or secondMaxPoint == -1:
        return secondMaxPoint
    return SecondMaxPoint(secondMaxPoint, borderPoint, curv, n-1)

# borderPoint={} 
# visitedPoint={}
# maximum = np.argmax(curv)
# visitedPoint[f'{maximum}']=maximum
# FindNeighbors(visitedPoint, borderPoint, 1.5, curv[maximum])

# point1 = SecondMaxPoint(maximum, borderPoint, curv, 1) 
# visitedPoint[f'{point1}']=point1
# FindNeighbors(visitedPoint, borderPoint, 1.5, curv[maximum])

# point2 = SecondMaxPoint(maximum, borderPoint, curv, 2) 
# visitedPoint[f'{point2}']=point2
# FindNeighbors(visitedPoint, borderPoint, 1.5, curv[maximum])

# point3 = SecondMaxPoint(maximum, borderPoint, curv, 11)
# visitedPoint[f'{point3}']=point3
# FindNeighbors(visitedPoint, borderPoint, 1.5, curv[maximum])

# point4 = SecondMaxPoint(maximum, borderPoint, curv, 11) #12
# visitedPoint[f'{point4}']=point4
# FindNeighbors(visitedPoint, borderPoint, 1.5, curv[maximum])

# point5 = SecondMaxPoint(maximum, borderPoint, curv, 15) #11
# visitedPoint[f'{point5}']=point5
# FindNeighbors(visitedPoint, borderPoint, 1.5, curv[maximum])

# point6 = SecondMaxPoint(maximum, borderPoint, curv, 1) #10
# visitedPoint[f'{point6}']=point6
# FindNeighbors(visitedPoint, borderPoint, 1.7, curv[maximum])

# point7 = SecondMaxPoint(maximum, borderPoint, curv, 2) #10
# visitedPoint[f'{point7}']=point7
# FindNeighbors(visitedPoint, borderPoint, 1.5, curv[maximum])

# # point8 = SecondMaxPoint(maximum, borderPoint, curv, 1) #10
# # visitedPoint[f'{point8}']=point8
# # FindNeighbors(visitedPoint, borderPoint, 1.5, curv[maximum])

borderPoint0={} 
visitedPoint0={}
maximum = np.argmax(curv)
visitedPoint0[f'{14969}']=14969
FindNeighbors(visitedPoint0, borderPoint0, 1.55, curv[maximum])

borderPoint1={} 
visitedPoint1={}
visitedPoint1[f'{315}']=315
FindNeighbors(visitedPoint1, borderPoint1, 1.0, curv[maximum])

borderPoint2={} 
visitedPoint2={}
visitedPoint2[f'{15064}']=15064
FindNeighbors(visitedPoint2, borderPoint2, 1.2, curv[maximum])

borderPoint3={} 
visitedPoint3={}
visitedPoint3[f'{28864}']=28864
FindNeighbors(visitedPoint3, borderPoint3, 1.4, curv[maximum])

borderPoint4={} 
visitedPoint4={}
visitedPoint4[f'{19509}']=19509
FindNeighbors(visitedPoint4, borderPoint4, 1.3, curv[maximum])

borderPoint5={} 
visitedPoint5={}
visitedPoint5[f'{26313}']=26313
FindNeighbors(visitedPoint5, borderPoint5, 1.3, curv[maximum])

borderPoint6={} 
visitedPoint6={}
visitedPoint6[f'{21469}']=21469
FindNeighbors(visitedPoint6, borderPoint6, 1.6, curv[maximum])

borderPoint7={} 
visitedPoint7={}
visitedPoint7[f'{4255}']=4255
FindNeighbors(visitedPoint7, borderPoint7, 1.4, curv[maximum])

def Plot():
    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True, scalars=curv, clim=[cmin, cmax])
    for i in borderPoint0.values():
        p.add_mesh(mesh.points[i], color="red", point_size=10)
    # for i in borderPoint1.values():
    #     p.add_mesh(mesh.points[i], color="blue")
    for i in borderPoint2.values():
        p.add_mesh(mesh.points[i], color="white", point_size=10)
    for i in borderPoint3.values():
        p.add_mesh(mesh.points[i], color="purple", point_size=10)
    for i in borderPoint4.values():
        p.add_mesh(mesh.points[i], color="orange", point_size=10)
    for i in borderPoint5.values():
        p.add_mesh(mesh.points[i], color="pink", point_size=10)
    for i in borderPoint6.values():
        p.add_mesh(mesh.points[i], color="brown", point_size=10)
    for i in borderPoint7.values():
        p.add_mesh(mesh.points[i], color="blue", point_size=10)
    p.show(cpos="yx")

Plot()
