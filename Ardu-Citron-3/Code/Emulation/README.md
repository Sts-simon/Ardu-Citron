# Ardu-Citron — émulation logicielle intégrée

Cette émulation *software-in-the-loop* relie le modèle d'avion, IMU/RC/caméra
synthétiques, le vrai détecteur ArUco du dépôt, le codec Python Pi et les modules
C++ réels du contrôleur ESP32 (Kalman, PID, mixage, limites servo, modes et
surveillance IMU) compilés dans une bibliothèque native. Aucun GPIO, servo réel
ni moteur n'est commandé.

## Lancer

Python 3.10+, g++, OpenCV contrib, NumPy et Matplotlib sont requis :

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python simulator.py
```

Une sortie plus courte peut être exécutée avec :

```sh
python simulator.py --duration 5 --out output-smoke
python simulator.py --config aircraft.json --out output-essai
```

Chaque exécution crée :

- `flight_log.csv` : une ligne par pas de simulation (pas configuré, 20 ms par
  défaut), avec position N/E/altitude, vitesses, airdata, attitude, vitesses de
  rotation, accélérations, IMU bruitée, RC, modes, état de santé, sorties PID,
  impulsions et débattements servo, forces/moments, résultats de vision et
  compteurs UART ;
- `flight_summary.png` : trajectoire, altitude/vitesse, attitude, commandes,
  sorties servo et dernière image virtuelle ;
- `summary.json` : seed, durée, état terminal, paramètres de haut niveau et
  avertissement de fidélité.

La seed déterministe est dans `aircraft.json`. Les scénarios `rc_command` y
commandent roll/pitch (degrés) et taux de lacet (degrés/s); le simulateur met
l'armement RC à l'état haut après chaque point de scénario. Chaque pas de la
simulation appelle les vrais filtres et contrôleur natifs. Le Pi voit des
images synthétiques via le détecteur ArUco existant; les poses sont encodées et
décodées à travers le codec UART v1. Le repère de la pose camera est encore
indicatif: la mesure n'est pas injectée comme consigne ESP32.

## Modèle et paramètres provisoires

`aircraft.json` commence avec un planeur d'1 m d'envergure : masse 0,55 kg,
surface alaire 0,18 m², corde 0,18 m et moments d'inertie grossièrement estimés.
Ces valeurs servent uniquement à faire tourner le pipeline. Le modèle est un
corps rigide 6-DOF en axes NED/body, avec coefficients aérodynamiques
quasi-stationnaires, polaires linéaires simples, traînée induite et commandes
directes des gouvernes. Ce n'est ni un modèle de profil mesuré, ni un modèle
CFD, ni un modèle certifié. Les coefficients, les sens de commandes et le
couplage servo/gouverne sont des hypothèses à remplacer par les données de
l'avion et à identifier expérimentalement.

La poussée est forcée à zéro. Le vol commence avec vitesse et altitude
initiales configurées et s'arrête au contact du sol. Le modèle de caméra place
des marqueurs statiques au sol, projette leur perspective, simule bruit,
flou, rolling shutter, distorsion et pertes de trame. Les effets et
l'orientation caméra sont encore simplifiés. La vérité terrain fournit position
et taille du marqueur, mais le codec actuel transmet des coordonnées image et
distance à la place d'une pose caméra résolue : ce point doit évoluer avant
toute fusion vision/attitude.

Les sorties servo/angles sont celles du contrôleur actuel puis converties en
angles aérodynamiques par un gain provisoire configurable. Les consignes de vol
ne doivent pas être considérées comme validées. La perte RC ou IMU déclenche
les modes logiciels existants, mais la stratégie de retour des gouvernes sur
un avion réel reste volontairement non affirmée : `centre` ici est seulement
le comportement actuel du firmware, pas une recommandation de sécurité en vol.

## Défauts simulables

Dans la section `sensors`, augmenter `imu_dropout_probability`,
`rc_dropout_probability` ou `uart_corruption_probability` (entre 0 et 1) pour
injecter des pertes d'échantillon/trame. `camera.drop_probability` simule une
image sans détection. Vent et bruit sont également configurables. Une seed fixe
rend ces cas reproductibles.

## À remplacer quand l'avion est mesuré

Masse prête à voler, envergure/surface/corde, position du centre de gravité,
matrice d'inertie complète, profils et polaires, vitesse de décrochage, limites
mécaniques et délais servo, gain des gouvernes, poussée/ESC/hélice plus tard,
matrice de montage IMU, position/orientation/intrinsèques de caméra, taille et
coordonnées des marqueurs, vent/densité, timings réels et calibration des
capteurs. Le `summary.json` rappelle que ce résultat n'est pas une validation
de vol.
