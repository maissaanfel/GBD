import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.classes import graph
#from numpy import inf
from IPython.display import display
from math import inf
import sys



def BellmanFord(graphe, origine):
    """
    Fonction qui permet de lancer l'algorithme de Bellman-Ford sur un graphe:
    - graphe : La Matrice du graphe
    - origine : Le sommet à partir duquel seront calculé toutes les distances
    L'algorithme de BellmanFord tolère les arcs de poids négatifs et détecte les
    cycles négatifs.
    """

    distances = [float("Inf")] * len(graphe)
    distances[origine] = 0

    for _ in range(len(graphe) - 1):
        for u in range(len(graphe)):
            for v in range(len(graphe)):
                if distances[u] + graphe[u][v] < distances[v]:
                        distances[v] = distances[u] + graphe[u][v]

    # Si on trouve une distance inférieure après le premier parcours, il y a
    # un cycle négatif
    for u in range(len(graphe)):
        for v in range(len(graphe)):
            if distances[u] + graphe[u][v] < distances[v]:
                print("Le graphe contient un cycle négatif")
                return

    return distances

if __name__ == '__main__':

    print("*** Bienvenue sur notre application ***")
    print("1.Graphes valués")
    print("2.Probléme du plus court chemin :")
    print("2.1 Algorithme de Bellman")
    print("2.2 Algorithme de Djikstra")
    print("2.3 Application ordonnancement")
    print("3 Algorithme de prime")
    C= float(input("Veuillez faire votre choix s'il vous plait: \n"))
    
if(C==2.1) :
    def np_inf(l):
      graphe = np.zeros( (l, l) )
      b = graphe.shape
      for i in range(b[0]):
          for j in range(b[0]):
              graphe[i][j] = inf
      return graphe

    l = int(input("Entrez le nombre de sommets "))
    sommets = input("Entrez les noms de vos sommets ").split(',')
    graphe = np_inf(l)
    print(graphe)
    no_d_aretes=int(input("Entrez le nombre d'aretes "))

    arete_avec_poids=[]

    for i in range(no_d_aretes):
        b = input("arete \n :"+str(i)).split(',')
        s1 = b[0]
        s2 = b[1]
        w = b[2]
        graphe[int(s1)][int(s2)] = int(w)
        display(graphe)
    
    distances = BellmanFord(graphe, 0)
    print("Distances depuis la Source ")

    for i in range(len(distances)):
        print("Sommet",i,"\t-->\t",distances[i])

elif(C== 3) :
    G = nx.Graph()
    matrice_G=[]
    N = int(input("Veuillez introduire le nombre de sommets du graphe : "))
    print() #améliorer l'affichage des requetes//////////////

#Creer une matrice de taille NxN contenant les valeurs que l'utilisateur aura choisi
    print("Veuillez introduire :") #améliorer l'affichage des requetes////////////// 

    for i in range(0, N):
        Poids=[]
        print() #améliorer l'affichage des requetes//////////////
    
  #remplir les poids entre le sommet actuel et tous les sommets de G
        for j in range(0, N):
            val=int(input("      le poids entre le sommet "+ str(i+1)+" et "+str(j+1)+" : "));
            Poids.append(val)
            if(not (val==0)): #si le poids n'est pas nul l'ajouter au graphe
                G.add_edge(i+1, j+1,weight=val)

        matrice_G.append(Poids)


    pos = nx.spring_layout(G)

    options = {
    "font_size": 20,
    "node_size": 1500,
    "node_color": '#c6baff',
    "edgecolors": "black",  
    "linewidths": 1,
    "width": 1,
    }


    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    nx.draw_networkx(G,pos, **options)

    ax = plt.gca()
    ax.margins(0.10)
    plt.axis("off")
    ax.set_title('G: Graphe original')

    plt.show()

#----------------------------Appliquer l'algorithme de prim sur G------------------------------

#Creer un vecteur pour marquer si le sommet a deja été selectionné ou pas (s'il fait partit des sommet de l'arbre couvrant de plus petit poids ou non)
    liste_sommets_selectionnes = []
    for i in range(1, N+1):
        liste_sommets_selectionnes.append(0) #initialement la liste contient N (nombre de sommets) valeurs mise a false==0 (aucun sommet n'est selectionné)

    nbr_arretes = 0

    liste_sommets_selectionnes[0] = True #selectionner le premier sommet de G pour demarrer avec l'algorithme de prim


    G2 = nx.Graph() #generer l'arbre couvrant

    #le nombre d'arretes dans un arbre couvrant de poids minimal est toujours inferieur au nombre de sommets - 1 soit : N-1
    while (nbr_arretes < N - 1):
    
        minimum = 9999999 #initialement on attribut au minimum une grande valeur
        sommet_a = 0
        sommet_b = 0

        for sommet_m in range(N): #parcourrir tous les sommets
            if liste_sommets_selectionnes[sommet_m]: #si le sommet n'est pas deja selectionné
            
                for sommet_n in range(N): #comparer les poids entre le sommet_m avec tous les autres sommets
                  if ((not liste_sommets_selectionnes[sommet_n]) and matrice_G[sommet_m][sommet_n]):  #si le sommet_n n'est pas deja selectionné et si le poids entre sommet_m et sommet_n n'est pas nul (il existe une arrete)
                    
                    if minimum > matrice_G[sommet_m][sommet_n]: #selectionner le sommet_n dont l'arrete est de poids minimum
                        minimum = matrice_G[sommet_m][sommet_n]
                        sommet_a = sommet_m
                        sommet_b = sommet_n

        G2.add_edge(sommet_a+1, sommet_b+1,weight=matrice_G[sommet_a][sommet_b]) #ajouuter les sommets ainsi que leurs poids à l'abre 
        liste_sommets_selectionnes[sommet_b] = True #marquer le sommet comme étant selectionné
        nbr_arretes += 1 #auguementer le nbr d'arretes de l'arbre couvrant de poids min
    pos2 = nx.spring_layout(G2)
    options2 = {
    "font_size": 20,
    "node_size": 1500,
    "node_color": 'white',
    "edgecolors": "#7558fc",  
    "linewidths": 1,
    "width": 1,
    }
    labels2 = nx.get_edge_attributes(G2,'weight')
    nx.draw_networkx_edge_labels(G2,pos2,edge_labels=labels2)

    nx.draw_networkx(G2,pos2, **options2)

    ax2 = plt.gca()
    ax2.margins(0.10)
    plt.axis("off")
    ax2.set_title('G2: Arbre couvrant de poids minimal de G')
    plt.show()

elif(C==2.3) :
    def bellmanlong(M, sommetDepart):
        distances = {} 
        predecesseurs = {}
        for sommet in M:
            distances[sommet] = -inf
            predecesseurs[sommet] = None
        distances[sommetDepart] = 0

        for i in range(len(M)-1):
            for j in M:
                for k in M[j]: 
                    if  distances[j] + M[j][k] > distances[k] :
                        distances[k]  = distances[j] + M[j][k]
                        predecesseurs[k] = j         
                
        return distances, predecesseurs

    G = nx.DiGraph()

#-------------------Matrice de poids qui correspond au projet--------------------------

    G.add_weighted_edges_from([(0, 1, 0.0), (0, 3, 5.0), (1, 3, 7.0), (1, 2, 12.0), (3, 2, 4.0), (2, 4, 6.0)])
    matrice_G = {'0':{'1':0.0,'3':5.0},'1':{'3':7.0,'2':12.0},'2':{'4':6.0},'3':{'2':4.0},'4':{}}

#---------------------Graphe potentiel-tache du projet--------------------------

    pos = nx.spring_layout(G)

    options = {
      "font_size": 20,
      "node_size": 800,
      "node_color": '#c6baff',
      "edgecolors": "black",  
      "linewidths": 1,
      "width": 1,
        }      
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    nx.draw_networkx(G,pos, **options)

    ax = plt.gca()
    ax.margins(0.28)
    plt.axis("off")
    ax.set_title('Potentiel_tache')
    plt.show()

#------------------Dates de début au plus tot de chacune des taches--------------------------
#---------Calcul des chemins les plus long depuis le sommet 0 à tous les autres--------------

    distances, predecesseurs = bellmanlong(matrice_G,'0')
    print("Date début au plus tot de chacune des taches :")
    for v in distances: print(str(v) + ' -> ' + str(distances[v]))

elif(C== 1):
    G = nx.Graph()
    MV=[]

#-------------------Matrice de poids--------------------------

#Demander a l'utilisateur d'introduire le nombre de sommets souhaités (N)
    N = int(input("Veuillez introduire le nombre de sommets du graphe : "))
    print()

#Creer une matrice de taille NxN contenant les valeurs que l'utilisateur aura choisi
    print("Veuillez introduire :")

    for i in range(0, N): #Pour chaque ligne de la matrice
      Poids=[]
      print() 
    
  #remplissage des poids entre le sommet actuel et le reste des sommets du graphe 
    for j in range(0, N): #Pour chaque colonne de la matrice
      val=int(input("      le poids entre le sommet "+ str(i)+" et "+str(j)+" : "));
      G.add_edge(i, j,weight=val)
      Poids.append(val)
    MV.append(Poids)


#-------------------Construction d'un graphe valué--------------------------

    pos = nx.spring_layout(G)
    options = {
        "font_size": 20,
        "node_size": 1500,
        "node_color": '#c6baff',
        "edgecolors": "black",  
        "linewidths": 1,
        "width": 1,
    }
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    nx.draw_networkx(G,pos, **options)
    ax = plt.gca()
    ax.margins(0.30)
    plt.axis("off")
    ax.set_title('G: Graphe valué')

    #-------------------Affichage du graphe valué--------------------------
    plt.show()

elif(C==2.2) :
    class Graph():

        def __init__(self, vertx):
            self.V = vertx
            self.graph = [[0 for column in range(vertx)]
                      for row in range(vertx)]
        def pSol(self, dist):
            for node in range(self.V):
                print("Distance depuis la source vers le sommet ", node, ":", dist[node])

        def minDistance(self, dist, sptSet):

            min = sys.maxsize

            for v in range(self.V):
                if dist[v] < min and sptSet[v] == False:
                    min = dist[v]
                    min_index = v

            return min_index

        def dijk(self, source):

                dist = [sys.maxsize] * self.V
                dist[source] = 0
                sptSet = [False] * self.V

                for cout in range(self.V):

                    u = self.minDistance(dist, sptSet)

                    sptSet[u] = True

                    for v in range(self.V):
                        if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]:
                            dist[v] = dist[u] + self.graph[u][v]

                self.pSol(dist)

def np_inf(l):
    M = np.zeros( (l, l) )
    b = M.shape
    for i in range(b[0]):
        for j in range(b[0]):
            M[i][j] = inf
    return M

l = int(input("Entrer le nombre de sommets"))
f = Graph(l)
sommets = input("entrer le nom des sommets").split(',')
M = np_inf(l)
print(M)
no_d_aretes=int(input("enter le nombre d'aretes"))

arete_avec_poids=[]

for i in range(no_d_aretes):
    b = input("arete \n"+str(i)).split(',')
    s1 = b[0]
    s2 = b[1]
    w = b[2]
    M[int(s1)][int(s2)] = int(w)

f.graph = M
f.dijk(0)
