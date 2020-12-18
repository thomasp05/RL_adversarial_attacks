# RL_adversarial_attacks

Projet final pour le cours d'apprentissage par renforcement (IFT-7201)

Présenté par Joel Vieira et Thomas Philippon

## Comment reproduire les résultats:  

Il y a trois scripts python dans le dossier **src** pour les expérimentations : 
 1. main_untargeted.py pour les untargeted attacks.
 2. main_targeted.py pour les targeted attacks
 3. main_one_NN_per_class.py pour les attaques avec un réseau par classe

Ces scripts peuvent être exécutés pour reproduire nos expérimentations à condition que la fonction de reward soit configurée proprement dans le fichier *environment.py* et que le nom du dossier où les résultats seront sauvegardés soit configuré. Voici les étapes à suivre pour reproduire nos expérimentations. Ces étapes sont identiques pour n'importe quel type d'attaque.

1. La première étape est de configurer le dossier ou les résultats seront sauvegardés. Cette étape se fait dans le script spécifique à l'attaque que l'on veut performer (un des trois scripts mentionnés plus haut). Comme on peut le voir dans la figure ci-dessous, il faut spécifier le nom du dossier (dans cet example c'est **untargetedattacks_file**). Il faut ensuite manuellement créer ce dossier dans le root directory du projet et créer les sous-dossiers *results*, *plots*, et *models* dans le dossier que l'on vient de créer. Le dossier *results* va contenir des fichier pdf avec les adversaires générés pour chaque épisode (les images des adversaires sont sauvegardées dans des fichiers pdf). Le dossier *plots* va contenir les figures pour le cumulative reward et loss en fonction du nombre d'épisode. Finalement, le dossier *models* va contenir le modèle .pt du réseau actor.  
![capture_](https://user-images.githubusercontent.com/25388214/102563062-7e247780-40a6-11eb-9e64-c35dcb677593.PNG)

2. La deuxième étape est de configurer la fonction de reward que l'on veut tester dans le script *environment.py*. Voici les 5 fonctions de rewards présentées dans le rapport ainsi que les lignes de code à *uncomment* pour chaque fonction de reward. 
   - R1 : Ligne 138 sur la figure ci-dessous
   - R2 : Ligne 138 et 160 sur la figure ci-dessous
   - R3 : Ligne 141 et sur la figure ci-dessous
   - R4 : Ligne 141 et 160 sur la figure ci-dessous
   - R5 : Ligne 144 à 153 sur la figure ci-dessous

![rewardfunction](https://user-images.githubusercontent.com/25388214/102563238-d0659880-40a6-11eb-8d8c-23f5382f7db2.PNG)

3. Quand les étapes 1 et 2 sont terminées, il reste à exécuter le script de l'attaque et d'observer les résultats dans le dossier créé à l'étape 1. 

## Où sont les résultats présentés dans le rapport:
Les résultats présentés dans notre rapport ont été placés dans trois dossiers différents. La structure des résultats est présentée dans les trois figures suivantes. 

1. Pour les untargeted attacks, les résultats sont dans le dossier *Untargeted_Attacks_Results*. 
![untargeted](https://user-images.githubusercontent.com/25388214/102567460-8e8d2000-40af-11eb-8c52-fee3d0280470.PNG)

2. Pour les untargeted attacks, les résultats sont dans le dossier *Targeted_Attacks_Results*
![targeted](https://user-images.githubusercontent.com/25388214/102567455-8cc35c80-40af-11eb-9117-7f3229499068.PNG)

3. Pour les untargeted attacks, les résultats sont dans le dossier *One_NN_per_Class_Results*
![one_per_class](https://user-images.githubusercontent.com/25388214/102567450-8b922f80-40af-11eb-9cbc-2fbbc5e8fc94.PNG)

