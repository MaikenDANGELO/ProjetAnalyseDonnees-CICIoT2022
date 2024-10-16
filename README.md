# Projet Data Set

Ce projet a pour objectif l'analyse de données tirées d'un dataset CIC IoT DataSet 2022, trouvé sur [https://www.unb.ca/cic/datasets/iotdataset-2022.html](www.unb.ca), dans une optique de cybersécurité.
Nous sommes une équipe de deux étudiants :
- Maïken D'Angelo
- Xavier Gousset

L'IoT (Internet of Things) ou Internet des Objets est tout ce qui concerne les objets connectés et la domotique. C'est un concept de plus en plus répandu dans notre société, ce qui rend notre quotidien d'autant plus vulnérable aux attaques de piratages ou d'intrusion.
Il est donc important de savoir analyser ces attaques pour mieux s'en protéger.

## Jeu de données

Le jeu de données fourni comprend 28 appareils différents, lesquels ont été testés 3 fois dans 10 modes différents. Ce qui donne 840 fichiers CSV différents, il faut donc regrouper les données de manières adéquate pour pouvoir les analyser facilement.

## Comment charger les données ?

Avec autant de fichiers CSV, il est impensable de tous les charger et analyser un à un. C'est pourquoi nous avons écrit un script permettant de parcourir tous les dossiers et fichiers, et de les charger dans un seul dataframe. De plus nous ajoutons à chaque fichier une nouvelle colonne permettant de savoir de quel apparail les données proviennent, et une autre pour savoir sous quelles conditions étaient cet appareil.

## Quelles données ? Quels traitements ?

Tout d'abord il faut nettoyer les données, par exemple en indexant correctement celles-ci, et en enlevant les valeurs nulles ou dupliquées. Après cela nous avons effectuer quelques tests pour prendre en main le jeu de données.

Puis nous avons commencé à décrire ces données, et trouver les colonnes interessantes / suspectes. Pour cela nous nous avons donner un aperçu en construisant une correlation heatmap, ce qui ne nous a pas beaucoup avancé. Ensuite nous avons d'utiliser un algorithme pour trouver les "most important features", avec lesquelles nous avons construit un pairplot d'après le type des appareils (home_automation, camera, audio), mais les données n'étaient pas très explicites.

C'est alors que nous avons commencé à entraîner des modèles de Machine Learning (ML), en commençant par un RandomForestClassifier, sur ces "most important features", afin de trouver le pourcentage de précision du modèle.

## Analyse de données

## Conclusion