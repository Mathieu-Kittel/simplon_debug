# Debugg d'application

Vous disposez d’une application Flask de classification d’images satellite (4 classes : desert, forest, meadow, mountain).

Un problème s'est glissé dans le code, à vous de le débusquer, trouvez l’anomalie, expliquez clairement la cause, proposez et appliquez un correctif minimal.

Documentez bien la résolution et expliquez ce qui n'allait pas, proposez des tests automatisés lors du déploiement CI/CD (comme on a vu avec github actions) pour que le problème détecté, et d'autres éventuels, ne mettent pas à mal l'application flask

Mettez en place une politique de journalisation et un alerting, quelques ressources :

- https://medium.com/@briankworld/logging-in-flask-introduction-and-practical-example-d2eeac0078b0
- https://last9.io/blog/flask-logging/
- https://betterstack.com/community/guides/logging/how-to-start-logging-with-flask/
- https://flask-fr.readthedocs.io/logging/
- https://flask.palletsprojects.com/en/stable/logging/


Proposez la possibilité de monitorer l'application grâce à flask_monitoring-dashboard :
- https://flask-monitoringdashboard.readthedocs.io/en/latest/


Mettez en place une feedback loop : récupérez le feedback de l'utilisateur (sur la page de résultat des boutons sont inactifs, l'utilisateur peut alors classifier lui-même l'image soumise) et réflechissez à comment l'intégrer à l'application. Schématisez via un diagramme fléché, proposez une modélisation d'une base de données recueillant les feedbacks, dans notre cas de figure, il faut récupérer trois éléments : l'image soumise, la prédiction faite par le modèle et le feedback de l'utilisateur (la classe choisie par l'utilisateur).

Expliquez comment un réentraînement peut être mis en place pour améliorer le modèle, justifiez en fournissant des sources.

Critères de réussite (checklist) :

- [ ] Le bug est identifié, expliqué et corrigé.
- [ ] Les tests attrapent l’ancien bug et passent en vert après correctif.
- [ ] La feedback loop enregistre correctement image + prédiction + label utilisateur.
- [ ] La CI GitHub Actions s’exécute à chaque PR et empêche l’intégration si un test échoue.