# Projet Data Set

Ce projet a pour objectif l'analyse de données tirées d'un dataset CIC IoT DataSet 2022, trouvé sur [https://www.unb.ca/cic/datasets/iotdataset-2022.html](www.unb.ca), dans une optique de cybersécurité.
Je forme ma propre équipe composée de moi-même:
- Maïken D'Angelo

L'IoT (Internet of Things) ou Internet des Objets est tout ce qui concerne les objets connectés et la domotique. C'est un concept de plus en plus répandu dans notre société, ce qui rend notre quotidien d'autant plus vulnérable aux attaques de piratages ou d'intrusion.
Il est donc important de savoir analyser ces attaques pour mieux s'en protéger.

## Jeu de données

Le jeu de données fourni comprend 28 appareils différents, lesquels ont été testés 3 fois dans 10 modes différents. Ce qui donne 840 fichiers CSV différents, il faut donc regrouper les données de manières adéquate pour pouvoir les analyser facilement.

## Comment charger les données ?

Avec autant de fichiers CSV, il est impensable de tous les charger et analyser un à un. C'est pourquoi j'ai écrit un script permettant de parcourir tous les dossiers et fichiers, et de les charger dans un seul dataframe. De plus on ajoute à chaque fichier une nouvelle colonne permettant de savoir de quel apparail les données proviennent, une autre pour savoir sous quelles conditions étaient cet appareil, et une dernière correspondant au type d'appareil.

## Quelles données ? Quels traitements ?

Tout d'abord il faut nettoyer les données, par exemple en indexant correctement celles-ci, et en enlevant les valeurs nulles ou dupliquées. Après cela j'ai effectué quelques tests pour prendre en main le jeu de données.

Puis j'ai commencé à décrire ces données, et trouver les colonnes interessantes / suspectes. Pour cela je me suis donné un aperçu en construisant une correlation heatmap, ce qui ne m'a pas beaucoup avancé. Ensuite j'ai utilisé un algorithme pour trouver les "most important features", avec lesquelles j'ai construit un pairplot d'après le type des appareils (home_automation, camera, audio), mais les données n'étaient pas très explicites.

C'est alors que j'ai commencé à entraîner des modèles de Machine Learning (ML), en commençant par un RandomForestClassifier, sur ces "most important features", afin de trouver le pourcentage de précision du modèle. En echaînant avec d'autres modèles comme GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier et ExtraTreesClassifier, j'ai pu comparer leur précision de différentes manières :
- en récupérant directement les valeurs et en faire un nouveau dataset afin de travailler dessus directement
- en utilisant un cross_val_score pour être sûre des valeurs et de même en faire un dataset.

On peut observer que les modèles donnent tous la même précision moyenne (au millième près), sauf AdaBoostClassifier, aux alentours de 91%.

Après cela, j'ai ajouté au dataframe les recall_score et f1_score de chaque models pour chaque most_important_features, et ai réalisé un graphique pour les comparer.

Pour vérifier mes résultats et mon workflow, j'ai décidé de lire l'article des scientifique ayant publié ce dataset et analyser ce qu'ils ont fait et comparer mon travail au leur, pour mieux comprendre et m'améliorer.

## Conclusion