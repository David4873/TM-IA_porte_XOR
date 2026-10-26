import numpy as np
from typing import Literal
import matplotlib.pyplot as plt

def run(entres: list, iteration: int, parametre: dict, mode: Literal["entrainement", "test", "executer"], alpha: float = 0.4):
    ## Initialisation des variables
    # Entrées du modèle
    x1 = entres[0]
    x2 = entres[1]

    # premiére couche
    w1 = parametre["w1"] #[w11, w12, w21, w22]
    b1 = parametre["b1"] #[b1, b2]
    o1 = "RELU"
    z1 = [0.0, 0.0] #[z1, z2]
    a1 = [0.0, 0.0] #[a1, a2]

    # seconde couche
    w2 = parametre["w2"] #[w11, w12]
    b2 = parametre["b2"] #[b1]
    o2 = "Sigmoïde"
    z2 = [0.0] #[z1]
    a2 = [0.0] #[a1]

    # erreur (résultat de la fonction de cout)
    c = 0.0


    def propagation():
        nonlocal x1, x2, z1, a1, z2, a2

        ## Opération (propagation)
        # Premiére couches
        z1[0] = x1*w1[0] + x2*w1[1] + b1[0]
        a1[0] = z1[0] if z1[0]>0 else 0

        z1[1] = x1*w1[2] + x2*w1[3] + b1[1]
        a1[1] = z1[1] if z1[1]>0 else 0

        # Seconde couches
        z2[0] = a1[0]*w2[0]+a1[1]*w2[1]+b2[0]
        a2[0] = 1 / (1 + np.exp(-z2[0]))


    def cout(a_list, y_list):
        nonlocal c

        m = len(y_list)
        np_a = np.array(a_list)
        np_y = np.array(y_list)
        
        # Écrêtage pour éviter log(0) qui donne -inf
        epsilon = 1e-15
        np_a = np.clip(np_a, epsilon, 1 - epsilon)
        c = - (1 / m) * np.sum(np_y * np.log(np_a) + (1 - np_y) * np.log(1 - np_a))
        

    def apprentisage(yn):
        nonlocal x1, x2, z1, a1, z2, a2, w1, b1, w2, b2

        # Initialistion des gradients
        dz1 = [0.0, 0.0] #[z1, z2]

        dw1 = [0.0, 0.0, 0.0, 0.0] #[w11, w12, w21, w22]
        db1 = [0.0, 0.0] #[b1, b2]
        dw2 = [0.0, 0.0] #[w11, w12]
        db2 = [0.0] #[b1]


        # Rétropropagetion
        # Gradients Couche 2
        dz2 = a2[0] - yn
        db2[0] = dz2
        dw2[0] = dz2*a1[0]
        dw2[1] = dz2*a1[1]

        # Gradients Couche 1
        dz1[0] = dz2*w2[0]
        dz1[0] = dz1[0] if z1[0]>0 else 0  # Dérivée de ReLU
        dz1[1] = dz2*w2[1]
        dz1[1] =  dz1[1] if z1[1]>0 else 0  # Dérivée de ReLU

        dw1[0] = dz1[0]*x1
        dw1[1] = dz1[0]*x2
        dw1[2] = dz1[1]*x1
        dw1[3] = dz1[1]*x2

        db1[0] = dz1[0]
        db1[1] = dz1[1]


        # Decent de gradient
        # premiére couche
        w1[0] -= alpha*dw1[0] #[w11]
        w1[1] -= alpha*dw1[1] #[w12]
        w1[2] -= alpha*dw1[2] #[w21]
        w1[3] -= alpha*dw1[3] #[w22]

        b1[0] -= alpha*db1[0] #[b2]
        b1[1] -= alpha*db1[1] #[b2]

        # seconde couche
        w2[0] -= alpha*dw2[0] #[w11]
        w2[1] -= alpha*dw2[1] #[w12]

        b2[0] -= alpha*db2[0] #[b1]
        pass


    if mode == "entrainement":
        print("\n--- Phase d'entrainement ---")
        
        # Intialisation des données d'entrainement
        x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [[0], [1], [1], [0]]

        # Initialise la liste pour stocker les valeurs des coûts
        historique_cout = []

        # Boucle d'apprentisage
        for i in range(0, iteration):

            a = []

            # Boucles pour l'entrainer sur chaque cas d'entrainement
            for xn, yn in zip(x, y):
                # Entrées du modèle
                x1 = xn[0]
                x2 = xn[1]

                propagation()

                apprentisage(yn[0])

                a.append([a2[0]])

            cout(a, y)
            historique_cout.append(c)

            if i % 20 == 0:
                print(f"Itération {i} - Coût : {c}")
        
        print(f'paramétre : "w1": [{w1[0]}, {w1[1]}, {w1[2]}, {w1[3]}], "b1": [{b1[0]}, {b1[1]}], "w2": [{w2[0]}, {w2[1]}], "b2": [{b2[0]}]')

        mode = "test"

        # 3. Générer le graphique de l'évolution du cout en fonction du nombre d'itération
        plt.figure(figsize=(8, 5))
        plt.plot(range(iteration), historique_cout, label="Fonction de coût", color="blue")
        plt.title("Évolution du coût en fonction des itérations")
        plt.xlabel("Itérations")
        plt.ylabel("Coût (Loss)")
        plt.grid(True)
        plt.legend()
        plt.show() # Affiche la fenêtre avec le graphique

    if mode == "test":
        print("\n--- Phase de Test ---")

        # Intialisation des valeurs
        x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [[0], [1], [1], [0]]
        a = []

        for xn, yn in zip(x, y):
            # Entrées du modèle
            x1 = xn[0]
            x2 = xn[1]
    
            propagation()

            # Convertit la probabilité de la réponse en réponse absolue pour une meilleure lisibilité
            # Exemple : 0.56 -> 1; 0.16 -> 0; 0.5 -> 0; 0.93 -> 1
            prediction_binaire = int(a2[0] > 0.5)

            a.append([prediction_binaire])
    

        for i in range(len(x)):
            print(f"Entrée : {x[i]} -> Attendu : {y[i]} | Prédit : {a[i]}")

        # Calcul de la précision globale
        np_a = np.array(a)
        np_y = np.array(y)
        precision = np.mean(np_a == np_y) * 100
        print(f"\nPrécision finale du réseau : {precision}%")

        return

    if mode == "executer":
        print("\n--- Phase d'execution ---")

        propagation()

        # Convertit la probabilité de la réponse en réponse absolue pour une meilleure lisibilité
        # Exemple : 0.56 -> 1; 0.16 -> 0; 0.5 -> 0; 0.93 -> 1
        a2[0] = int(a2[0] > 0.5)

        print(f"Entrée : {[x1, x2]} ->  Prédit : {a2[0]}")

        return
    return


parametre_default = {
    # Première couche (2 neurones en entrée, 2 neurones cachés)
    "w1": [0.1, 0.6,  # [w11, w12] -> Poids connectés au premier neurone caché
        1.5, 0.3      # [w21, w22] -> Poids connectés au second neurone caché
    ],
    "b1": [10.1, 0.8],  # [b1, b2] -> Biiais des deux neurones cachés

    # Seconde couche (2 neurones cachés en entrée, 1 neurone de sortie)
    "w2": [1.0, 1.5   # [w11, w12] -> Poids connectés au neurone de sortie
    ],
    "b2": [1.6]        # [b1] -> Biais du neurone de sortie
}



run([1, 0], 500, parametre_default, "entrainement", 0.2)

