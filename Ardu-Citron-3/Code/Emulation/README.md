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

### Interface Python de bureau

Lancer l'interface native Tkinter (elle fait partie de Python, aucune dépendance
supplémentaire) :

```sh
python3 sim_ui.py
```

L'interface permet d'éditer la configuration complète en JSON (avion, météo,
capteurs, caméra, marqueurs, scénarios RC, navigation), de choisir la durée,
de lancer le simulateur et de consulter la courbe ainsi que les fichiers
produits. Les paramètres utilisés et résultats sont enregistrés dans
`output-ui/latest/`. L'Auto reste désactivé par défaut et aucun matériel n'est
commandé. Un environnement de bureau avec Tkinter est nécessaire pour afficher
la fenêtre; les tests de simulation restent exécutables sans interface avec
`python3 simulator.py`.

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
décodées à travers le codec UART v1.

## Navigation expérimentale et géorepérage

Le module partagé `../Pi/autonomy.py` fournit une première estimation de pose
par PnP ArUco, une propagation inertielle simplifiée corrigée par caméra, une
entrée GPS optionnelle et un guidage vers les marqueurs avec récupération vers
le centre si l'avion approche du rectangle. Le même code est importé par
l'émulateur et le runtime CM4. Rectangle, marge, marqueurs, calibration caméra
et sources de position se règlent dans ce fichier; dans la config de simulation,
`autopilot.enabled` reste à `false` par défaut pour n'appliquer que les
scénarios RC pendant les premiers tests. Le journal ajoute l'estimation de
navigation et les consignes de guidage pour permettre comparaison.

La configuration de démonstration propose un rectangle local de 200 × 200 m,
avec une marge intérieure de 30 m et quatre repères d'essai à remplacer par
l'arpentage réel. Le premier marqueur de l'émulateur est rapproché de son point
de départ uniquement pour vérifier la chaîne caméra→pose en test automatisé. Le
solveur rejette les poses dont l'erreur de reprojection dépasse 3 px ou dont
les points sont derrière la caméra. Une navigation périmée/incertaine produit
`NAV_UNCERTAIN` et aucune consigne. Le Pi peut encoder des consignes type 2,
mais `autopilot.enabled` reste `false` par défaut et l'ESP32 conserve également
`AUTO_MODE_ENABLED=false`; la transmission seule n'arme jamais le contrôleur.
Le Pi doit recevoir de vrais adaptateurs IMU/GPS avant que sa navigation puisse
être considérée valide. Le GPS simulé s'active via `navigation.use_gps`; IMU et
vision peuvent également être activées/désactivées. Cette fusion complémentaire
est un prototype, non une INS robuste.

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

## Runtime Raspberry Pi CM4

Depuis `Code/Pi`, installer `python3 -m pip install -r requirements.txt`,
calibrer la caméra et modifier `autonomy.json` (intrinsèques, matrice de
montage, emplacement et orientation de chaque repère et limites du rectangle).
Puis lancer `python3 autonomy_runtime.py --dry-run` pour afficher détections et
état de navigation sans transmission. Il faut fournir les adaptateurs réels
`imu_provider` et/ou `gps_provider` adaptés au matériel avant d'obtenir une
navigation valide.Le mode UART transmet les poses caméra comme télémétrie. Si et seulement si
`autopilot.enabled` est changé explicitement, le runtime émet également des
consignes type 2 toutes les 100 ms ou à chaque changement, valables 250 ms.
L'ESP32 vérifie le CRC, la séquence, les limites et la fraîcheur; AUTO exige en
plus l'autorisation compile-time `AUTO_MODE_ENABLED`, le sélecteur AUTO et
l'action armement côté RC. À désactiver tant que les adaptateurs de capteurs,
la calibration et les essais au sol ne sont pas faits. Aucune sortie Pi ne
commande directement servos ou moteur.


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
