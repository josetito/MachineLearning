import random as rand
import math as math
from point import Point
#import pkg_resources
#pkg_resources.require("matplotlib")
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class clustering:
    def __init__(self, geo_locs_, k_):
        self.geo_locations = geo_locs_
        self.k = k_
        self.clusters = []  #clusters of nodes
        self.means = []     #means of clusters
        self.debug = False  #debug flag

    #este metodo devuelve el siguiente nodo aleatorio
    def next_random(self, index, points, clusters):
        #Seleccione el siguiente nodo que tenga la distancia maxima de otros nodos
        dist = {}
        for point_1 in points:
            if self.debug:
                print 'point_1: %f %f' % (point_1.latit, point_1.longit) 
            #Calcular la distancia entre el nodo y todos los otros puntos en el grupo
            for cluster in clusters.values():
                point_2 = cluster[0]
                if self.debug:
                    print 'point_2: %f %f' % (point_2.latit, point_2.longit)
                if point_1 not in dist:
                    dist[point_1] = math.sqrt(math.pow(point_1.latit - point_2.latit,2.0) + math.pow(point_1.longit - point_2.longit,2.0))       
                else:
                    dist[point_1] += math.sqrt(math.pow(point_1.latit - point_2.latit,2.0) + math.pow(point_1.longit - point_2.longit,2.0))
        if self.debug:
            for key, value in dist.items():
                print "(%f, %f) ==> %f" % (key.latit,key.longit,value)
        #devolvemos el punto que tiene la distancia maxima de los nodos anteriores
        count_ = 0
        max_ = 0
        for key, value in dist.items():
            if count_ == 0:
                max_ = value
                max_point = key
                count_ += 1
            else:
                if value > max_:
                    max_ = value
                    max_point = key
        return max_point



    #Este metodo calcula los puntos iniciales
    def initial_means(self, points):
        #Escoger el primer nodo al azar
        point_ = rand.choice(points)
        if self.debug:
            print 'point#0: %f %f' % (point_.latit, point_.longit)
        clusters = dict()
        clusters.setdefault(0, []).append(point_)
        points.remove(point_)
        #Elegimos k-1 puntos aleatorios
        for i in range(1, self.k):
            point_ = self.next_random(i, points, clusters)
            if self.debug:
                print 'point#%d: %f %f' % (i, point_.latit, point_.longit)
            #clusters.append([point_])
            clusters.setdefault(i, []).append(point_)
            points.remove(point_)
        #Calcular media de clusters
        #self.print_clusters(clusters)
        self.means = self.compute_mean(clusters)
        if self.debug:
            print "initial means:"
            self.print_means(self.means)



    def compute_mean(self, clusters):
        means = []
        for cluster in clusters.values():
            mean_point = Point(0.0, 0.0)
            cnt = 0.0
            for point in cluster:
                #print "compute: point(%f,%f)" % (point.latit, point.longit)
                mean_point.latit += point.latit
                mean_point.longit += point.longit
                cnt += 1.0
            mean_point.latit = mean_point.latit/cnt
            mean_point.longit = mean_point.longit/cnt
            means.append(mean_point)
        return means




    #Este metodo asigna nodos al cluster con la menor media
    def assign_points(self, points):
        if self.debug:
            print "assign points"
        clusters = dict()
        for point in points:
            dist = []
            if self.debug:
                print "point(%f,%f)" % (point.latit, point.longit)
            #Encontrar el mejor cluster para este nodo
            for mean in self.means:
                dist.append(math.sqrt(math.pow(point.latit - mean.latit,2.0) + math.pow(point.longit - mean.longit,2.0)))
            #Encontrar la media mas pequena
            if self.debug:
                print dist
            cnt_ = 0
            index = 0
            min_ = dist[0]
            for d in dist:
                if d < min_:
                    min_ = d
                    index = cnt_
                cnt_ += 1
            if self.debug:
                print "index: %d" % index
            clusters.setdefault(index, []).append(point)
        return clusters



    def update_means(self, means, threshold):
        #Comprobar la media actual con la anterior para ver si debemos parar
        for i in range(len(self.means)):
            mean_1 = self.means[i]
            mean_2 = means[i]
            if self.debug:
                print "mean_1(%f,%f)" % (mean_1.latit, mean_1.longit)
                print "mean_2(%f,%f)" % (mean_2.latit, mean_2.longit)            
            if math.sqrt(math.pow(mean_1.latit - mean_2.latit,2.0) + math.pow(mean_1.longit - mean_2.longit,2.0)) > threshold:
                return False
        return True

    #funcion debug: imprimir puntos del cluster
    def print_clusters(self, clusters):
        cluster_cnt = 1
        for cluster in clusters.values():
            print "nodes in cluster #%d" % cluster_cnt
            cluster_cnt += 1
            for point in cluster:
                print "point(%f,%f)" % (point.latit, point.longit)

    #imprimir means
    def print_means(self, means):
        for point in means:
            print "%f %f" % (point.latit, point.longit)

    #k_means algorithm
    def k_means(self, plot_flag):
        if len(self.geo_locations) < self.k:
            return -1   #error
        points_ = [point for point in self.geo_locations]
        #calcular means iniciales
        self.initial_means(points_)
        stop = False
        while not stop:
            #Paso de asignacion: asignar cada nodo al cluster con la media mas cercana
            points_ = [point for point in self.geo_locations]
            clusters = self.assign_points(points_)
            if self.debug:
                self.print_clusters(clusters)
            means = self.compute_mean(clusters)
            if self.debug:
                print "means:"
                print self.print_means(means)
                print "update mean:"
            stop = self.update_means(means, 0.01)
            if not stop:
                self.means = []
                self.means = means
        self.clusters = clusters
        #plot cluster para la evaluacion
        if plot_flag:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            markers = ['o', 'd', 'x', 'h', 'H', 7, 4, 5, 6, '8', 'p', ',', '+', '.', 's', '*', 3, 0, 1, 2]
            colors = ['r', 'k', 'b', [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
            cnt = 0
            for cluster in clusters.values():
                latits = []
                longits = []
                for point in cluster:
                    latits.append(point.latit)
                    longits.append(point.longit)
                ax.scatter(longits, latits, s=60, c=colors[cnt], marker=markers[cnt])
                cnt += 1
            plt.show()
        return 0