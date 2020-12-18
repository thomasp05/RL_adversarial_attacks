# RL_adversarial_attacks

Projet final pour le cours d'apprentissage par renforcement (IFT-7201)

Présenté par Joel Vieira et Thomas Philippon

## Comment reproduire les résultats:  

Il y a trois scripts python dans le dossier **src** pour les expérimentations : 
 1. main_untargeted.py pour les untargeted attacks.
 2. main_targeted.py pour les targeted attacks
 3. main_one_NN_per_class.py pour les attaques avec un réseau par classe

Ces scripts peuvent être exécutés pour reproduire nos expérimentations à condition que la fonction de reward soit configurée proprement dans le fichier *environment.py* et que le nom du dossier ou les résultats seront sauvegardé soit configuré. 

![capture_](https://user-images.githubusercontent.com/25388214/102563062-7e247780-40a6-11eb-9e64-c35dcb677593.PNG)

![rewardfunction](https://user-images.githubusercontent.com/25388214/102563238-d0659880-40a6-11eb-8d8c-23f5382f7db2.PNG)

## Où sont les résultats présentés dans le rapport:

Les résultats présentés dans notre rapport ont été placés dans trois dossiers différents.

1. Pour les untargeted attacks, les résultats sont dans le dossier *Untargeted_Attacks_Results*
![untargeted](https://user-images.githubusercontent.com/25388214/102561756-829b6100-40a3-11eb-978d-7b8d5cb893d7.PNG)

2. Pour les untargeted attacks, les résultats sont dans le dossier *Targeted_Attacks_Results*

![targeted](https://user-images.githubusercontent.com/25388214/102561753-80d19d80-40a3-11eb-90c2-02e9660fc74c.PNG)

3. Pour les untargeted attacks, les résultats sont dans le dossier *One_NN_per_Class_Results*
![one_class_per_label](https://user-images.githubusercontent.com/25388214/102561747-7dd6ad00-40a3-11eb-8988-28319ab0c62b.PNG)

