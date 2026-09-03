# Contribuer

Créez une branche courte, limitez chaque commit à un changement cohérent et ouvrez une pull request vers `main`. Exécutez avant soumission :

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

Ne versionnez aucune donnée SAFRAN/SIM brute, clé, archive CUDA, prédiction complète ou environnement virtuel. Décrivez dans la pull request la période de données, le protocole et les sorties produites. Chaque membre doit pousser ses propres commits depuis son compte GitHub afin que l’historique reflète les contributions.
