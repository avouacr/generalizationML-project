# generalizationML-project

**Sujet :** En apprentissage supervisé classique, les méthodes de réduction de variance
(SAG, SVRG) combinent des updates de faible complexité (similaire à SGD) à
un taux de convergence rapide. Cependant, ces méthodes sont peu utilisées en
pratique en Deep Learning.

Le but de ce projet est de comparer le score de généralisation de différentes
méthodes d'optimisation dans le cadre d'une architecture simple de Deep
Learning pour une tache de vision. On pourra comparer par exemple SVRG,
SAG, SGD, GD, ainsi que tout autre algorithme d'optimisation que vous
souhaiterez utiliser (entropy-SGD, averaging, BP-SVRG).


**Données :** Le but est de travailler sur la base de données "CIFAR", si
nécessaire en sous samplant (1/10e des images par exemple).

**Implémentation :** Vous pouvez utiliser Keras, Pytorch ou TensorFlow ou autre.
Cependant, une implémentation correcte d'un réseau simple est disponible avec
Keras sur le moodle, je vous recommande de l'utiliser car elle constitue un
point de départ substantiel.

(Si nécessaire, vous pouvez utiliser Google Colab pour disposer d'un GPU).
