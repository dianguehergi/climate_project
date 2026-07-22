# Synthèse expérimentale des modèles Deep Learning

Dans cette partie du projet, nous avons progressivement étendu l’expérimentation Deep Learning depuis une prédiction locale sur un point de grille jusqu’à une modélisation globale sur l’ensemble des 9 892 points de grille SAFRAN disponibles.

La première étape a consisté à entraîner un modèle ConvLSTM sur 100 centres de grille avec un voisinage spatial complet 5×5. Cette expérience permettait de vérifier si l’exploitation d’un contexte spatial local pouvait améliorer la prédiction de la température à J+1. Le modèle a obtenu une MAE de 1,4236 °C et un RMSE de 1,8316 °C, soit une amélioration respective de 13,29 % et 14,14 % par rapport à la baseline de persistance.

Nous avons ensuite généralisé l’apprentissage à l’ensemble des 9 892 points de grille SAFRAN avec une configuration 1×1. Cette approche permet d’utiliser tous les points disponibles, mais sans exploiter explicitement les voisins spatiaux. Le modèle ConvLSTM global 1×1 obtient une MAE de 1,4317 °C, un RMSE de 1,8514 °C et un R² de 0,9287. Il améliore la persistance sur 9 877 points parmi les 9 892, ce qui montre une très bonne robustesse spatiale.

Afin d’exploiter à la fois l’ensemble de la grille et le contexte spatial local, nous avons développé un ConvLSTM 5×5 masqué. Le masque permet de gérer les points situés sur les bords du domaine, qui ne disposent pas toujours d’un voisinage complet. L’analyse de la grille montre que 8 415 points disposent d’un patch 5×5 complet, tandis que 1 477 points présentent un voisinage incomplet.

Le ConvLSTM 5×5 masqué obtient les meilleures performances globales. Sur l’évaluation complète des 21 386 504 séquences de test, il atteint une MAE de 1,4221 °C, un RMSE de 1,8326 °C et un R² de 0,9302. Il améliore la persistance de 13,57 % en MAE et de 14,22 % en RMSE. Comparé au modèle 1×1, il améliore la MAE sur 6 902 points et le RMSE sur 7 814 points.

L’analyse locale met toutefois en évidence un compromis. Le modèle 5×5 masqué présente 56 points non améliorés en MAE, contre 15 pour le modèle 1×1. Parmi ces 56 points, 43 sont associés à un patch spatial incomplet. Cela indique que le contexte spatial améliore les performances globales, mais peut introduire une instabilité locale lorsque l’information de voisinage est partielle.

Les points non améliorés du modèle 1×1 ont également été géolocalisés. Ils se concentrent principalement dans des zones méditerranéennes chaudes, notamment en Corse et dans le sud-est de la France. Leur température moyenne est de 14,82 °C, contre 9,97 °C pour l’ensemble de la grille. Ces points ne correspondent donc pas à des zones froides de montagne, mais plutôt à des zones chaudes où la persistance est déjà très performante et où le modèle présente un léger biais négatif.

Au final, le ConvLSTM 5×5 masqué est retenu comme modèle principal. Il offre le meilleur compromis entre performance globale, exploitation du contexte spatial et couverture complète de la grille SAFRAN. Le modèle 1×1 reste une référence utile, car il est légèrement plus stable localement sur certains points de bord ou zones à voisinage incomplet.
