# Synthèse expérimentale — Modélisation Deep Learning SAFRAN

Dans cette partie du projet, nous avons progressivement étendu l'entraînement des modèles Deep Learning depuis un point de grille isolé jusqu'à l'ensemble des 9 892 points de grille SAFRAN disponibles.

La première étape a consisté à établir une référence avec un modèle ConvLSTM entraîné sur 100 centres de grille avec un patch spatial complet 5×5. Ce modèle a obtenu une MAE de 1,4236 °C et un RMSE de 1,8316 °C, soit une amélioration de 13,29 % en MAE et de 14,14 % en RMSE par rapport à la persistance.

Nous avons ensuite entraîné un modèle ConvLSTM global sur l'ensemble des 9 892 points de grille en configuration 1×1. Cette approche permet d'utiliser tous les points disponibles, mais sans contexte spatial local. Le modèle a obtenu une MAE de 1,4317 °C, un RMSE de 1,8514 °C et un R² de 0,9287. Il améliore la persistance sur 9 877 points parmi les 9 892, ce qui montre une très bonne robustesse spatiale.

Afin d'intégrer le contexte spatial tout en conservant l'ensemble des points de grille, nous avons ensuite entraîné un ConvLSTM 5×5 masqué. Le masque permet de gérer les points situés sur les bords du domaine, qui ne possèdent pas toujours un voisinage 5×5 complet. Sur les 9 892 points, 8 415 disposent d'un patch 5×5 complet et 1 477 ont un patch incomplet.

Le modèle ConvLSTM 5×5 masqué obtient les meilleures performances globales, avec une MAE de 1,4221 °C, un RMSE de 1,8326 °C et un R² de 0,9302. Il améliore la persistance de 13,57 % en MAE et de 14,22 % en RMSE. Il est meilleur que le modèle 1×1 sur 6 902 points en MAE et sur 7 814 points en RMSE.

L'analyse locale montre cependant que le modèle 5×5 masqué présente 56 points non améliorés en MAE, contre 15 pour le modèle 1×1. Ces points sont majoritairement associés à des patchs incomplets : 43 des 56 points non améliorés en 5×5 ne disposent pas d'un voisinage spatial complet. Ce résultat montre que le contexte spatial améliore les performances globales, mais peut introduire une instabilité locale lorsque l'information de voisinage est partielle.

Au final, le ConvLSTM 5×5 masqué est retenu comme modèle principal, car il offre le meilleur compromis global entre précision, exploitation du contexte spatial et couverture complète de la grille SAFRAN.
