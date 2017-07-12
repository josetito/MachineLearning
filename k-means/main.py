import random as rand
from clustering import clustering
from point import Point
import csv

geo_locs = []
#loc_ = Point(0.0, 0.0)  #tupla
#geo_locs.append(loc_)
#Leer el archivo de entrada csv y almacene cada ubicacion de fuente como un objeto Point (latit, longit)
f = open('/home/jrarguedas/Documentos/IA/k-means/drinking_fountains.csv', 'r')
reader = csv.reader(f, delimiter=",")
for line in reader:
    loc_ = Point(float(line[0]), float(line[1]))  #tupla
    geo_locs.append(loc_)
print len(geo_locs)
for p in geo_locs:
    print "%f %f" % (p.latit, p.longit)
#run k_means clustering. El segundo prametro es el numero de clusters
cluster = clustering(geo_locs, 4)
flag = cluster.k_means(True)
if flag == -1:
    print "Error in arguments!"
else:
    #Los resultados de la agrupacion son una lista de listas en las que cada lista representa un grupo
    print "clustering results:"
    cluster.print_clusters(cluster.clusters)
    #cluster.k_means(flag)
