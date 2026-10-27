import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def run(entrées: list, iteration: int, paramètres: dict, mode: Literal["entrainement", "test", "executer"], taux_dapprentissage: float = 0.4, résultat_en_probabiliter: bool = False):
    ## Initialisation des variables
    # Entrées du modèle
    x1 = entrées[0]
    x2 = entrées[1]

    # première couche
    w1 = paramètres["w1"] #[w11, w12, w21, w22]
    b1 = paramètres["b1"] #[b1, b2]
    o1 = "RELU"
    z1 = [0.0, 0.0] #[z1, z2]
    a1 = [0.0, 0.0] #[a1, a2]

    # seconde couche
    w2 = paramètres["w2"] #[w11, w12]
    b2 = paramètres["b2"] #[b1]
    o2 = "Sigmoïde"
    z2 = [0.0] #[z1]
    a2 = [0.0] #[a1]

    # l'erreur (résultat de la fonction de coût)
    c = 0.0


    def propagation():
        nonlocal x1, x2, z1, a1, z2, a2

        ## Opération (propagation)
        # Première couches
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
        

    def apprentissage(yn):
        nonlocal x1, x2, z1, a1, z2, a2, w1, b1, w2, b2

        # Initialisation des gradients
        dz1 = [0.0, 0.0] #[z1, z2]

        dw1 = [0.0, 0.0, 0.0, 0.0] #[w11, w12, w21, w22]
        db1 = [0.0, 0.0] #[b1, b2]
        dw2 = [0.0, 0.0] #[w11, w12]
        db2 = [0.0] #[b1]


        # Rétropropagation
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


        # Descente de gradient
        # première couche
        w1[0] -= taux_dapprentissage*dw1[0] #[w11]
        w1[1] -= taux_dapprentissage*dw1[1] #[w12]
        w1[2] -= taux_dapprentissage*dw1[2] #[w21]
        w1[3] -= taux_dapprentissage*dw1[3] #[w22]

        b1[0] -= taux_dapprentissage*db1[0] #[b2]
        b1[1] -= taux_dapprentissage*db1[1] #[b2]

        # seconde couche
        w2[0] -= taux_dapprentissage*dw2[0] #[w11]
        w2[1] -= taux_dapprentissage*dw2[1] #[w12]

        b2[0] -= taux_dapprentissage*db2[0] #[b1]
        pass


    if mode == "entrainement":
        print("\n--- Phase d'entrainement ---")
        
        # Initialisation des données d'entrainement
        x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [[0], [1], [1], [0]]

        # Initialise la liste pour stocker les valeurs des coûts
        historique_cout = []

        # Boucle d'apprentissage
        for i in range(1, iteration+1):

            a = []

            # Boucles pour l'entrainer sur chaque cas d'entrainement
            for xn, yn in zip(x, y):
                # Entrées du modèle
                x1 = xn[0]
                x2 = xn[1]

                propagation()

                apprentissage(yn[0])

                a.append([a2[0]])

            cout(a, y)
            historique_cout.append(c)

            if i % 20 == 0 or i == 1:
                print(f"Itération {i} - Coût : {c}")
        
        print(f'paramètre : "w1": [{w1[0]}, {w1[1]}, {w1[2]}, {w1[3]}], "b1": [{b1[0]}, {b1[1]}], "w2": [{w2[0]}, {w2[1]}], "b2": [{b2[0]}]')

        mode = "test"

        # 3. Générer le graphique de l'évolution du coût en fonction du nombre d'itération
        plt.figure(figsize=(8, 5))
        plt.plot(range(iteration), historique_cout, label="Fonction de perte", color="blue")
        plt.title("Évolution de l'erreur en fonction du nombre d'itérations")
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("Erreur")
        plt.grid(True)
        plt.legend()
        plt.show() # Affiche la fenêtre avec le graphique

    if mode == "test":
        print("\n--- Phase de Test ---")

        # Initialisation des valeurs
        x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [[0], [1], [1], [0]]
        a_binaire = []
        a_probabiliste = []

        for xn, yn in zip(x, y):
            # Entrées du modèle
            x1 = xn[0]
            x2 = xn[1]
    
            propagation()

            # Convertit la probabilité de la réponse en réponse absolue pour une meilleure lisibilité
            # Exemple : 0.56 -> 1; 0.16 -> 0; 0.5 -> 0; 0.93 -> 1
            if résultat_en_probabiliter == False:
                prediction_binaire = int(a2[0] > 0.5)
                a_binaire.append([prediction_binaire])
            else:
                prediction_binaire = float(a2[0])
                a_probabiliste.append([prediction_binaire])

            
    

        for i in range(len(x)):
            if résultat_en_probabiliter == False:
                print(f"Entrée : {x[i]} -> Attendu : {y[i]} | Prédit : {a_binaire[i]}")
            else:
                print(f"Entrée : {x[i]} -> Attendu : {y[i]} | Prédit : {a_probabiliste[i]}")
                        

        # Calcul de la précision globale
        if résultat_en_probabiliter == False:
            np_a = np.array(a_binaire)
            np_y = np.array(y)
            precision = np.mean(np_a == np_y) * 100
            print(f"\nPrécision finale du réseau : {precision}%")
        else:
            np_a = np.array(a_probabiliste)
            np_y = np.array(y)
            
            # Calcul de l'AUC-ROC (prend les probabilités brutes et les vrais y)
            auc_score = roc_auc_score(np_y, np_a) * 100
            print(f"\nScore AUC-ROC final du réseau : {auc_score:.2f}%\nL'AUC-ROC est une méthode pour calculer la précision du modèle\nAller regarde pour plus de détaille sur le fonctionnement : https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=fr")


        return

    if mode == "executer":
        print("\n--- Phase d'execution ---")

        propagation()

        # Convertit la probabilité de la réponse en réponse absolue pour une meilleure lisibilité
        # Exemple : 0.56 -> 1; 0.16 -> 0; 0.5 -> 0; 0.93 -> 1
        if résultat_en_probabiliter == False:
            a2[0] = int(a2[0] > 0.5)

        print(f"Entrée : {[x1, x2]} ->  Prédit : {a2[0]}")

        return
    return


paramètres_default = {
    # Première couche (2 neurones en entrée, 2 neurones cachés)
    "w1": [0.1, 0.6,  # [w11, w12] -> Poids connectés au premier neurone caché
        1.5, 0.3      # [w21, w22] -> Poids connectés au second neurone caché
    ],
    "b1": [10.1, 0.8],  # [b1, b2] -> Biais des deux neurones cachés

    # Seconde couche (2 neurones cachés en entrée, 1 neurone de sortie)
    "w2": [1.0, 1.5   # [w11, w12] -> Poids connectés au neurone de sortie
    ],
    "b2": [1.6]       # [b1] -> Biais du neurone de sortie
}

# Coller les paramètres de votre modèle entre les < {} >.
paramètres_personnaliser = {

}

run([1, 0], 500, paramètres_default, "entrainement", 0.1, True)