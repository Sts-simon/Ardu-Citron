# Firmware ESP32 — mise en place logicielle

## Build reproductible

Le projet PlatformIO est défini par `platformio.ini` (ESP32 DevKit générique,
Arduino-ESP32 via PlatformIO, `ESP32Servo` épinglé). Depuis ce dossier :

```sh
python3 -m pip install -r requirements-dev.txt
pio run -e esp32
```

L'environnement matériel est déclaré ici comme `esp32dev`; confirmer le modèle
exact de carte avant le premier flash. Les tests natifs indépendants du matériel
s'exécutent avec `test/native/run.sh` (g++ requis). Ils ne remplacent pas le build
ESP32 ni les essais du câblage.

## Calibration, surveillance et modes

La calibration collecte jusqu'à 400 échantillons sur 10 secondes et échoue si
moins de 360 sont valides. Les erreurs I2C consécutives et l'absence de nouvelles
lectures IMU au-delà du délai fixé passent le moniteur IMU en défaut mémorisé.
Un défaut capteur interdit l'armement. La machine d'états distingue démarrage,
désarmé, stabilisation, failsafe et défaut capteur. La sortie moteur reste hors
de ce firmware tant que le matériel/ESC n'est pas défini.

L'armement n'est permis qu'après initialisation valide, commande récente et
trame RC valide récente. Le timeout de commande ou RC fait entrer en FAILSAFE,
qui nécessite un désarmement explicite avant une nouvelle demande d'armement.
Cette transition logicielle ne définit PAS la position des gouvernes à adopter
en vol : celle-ci reste à décider et à valider sur l'aéronef. Ne pas considérer
les valeurs actuelles de PID, trims, sens ou débattements comme validées.

## UART Pi → ESP32 (version 1)

L'interface est UART2, RX GPIO16 / TX GPIO17, 115200 bauds, 8N1. Relier TX Pi
au RX ESP32 et RX Pi au TX ESP32 avec une masse commune et des niveaux logiques
3,3 V. UART0 reste libre pour le flash/debug. Vérifier le brochage de la carte
avant connexion.

La trame est fixe (41 octets), sans padding C/C++ :

| Offset | Taille | Champ |
|---:|---:|---|
| 0 | 2 | Sync `A5 5A` |
| 2 | 1 | Version `1` |
| 3 | 1 | Type pose `1` |
| 4 | 1 | Taille payload `34` |
| 5 | 2 | Numéro de séquence `uint16`, little-endian |
| 7 | 4 | Timestamp émetteur `uint32 ms`, little-endian (horloge Pi) |
| 11 | 28 | `x,y,z,roll,pitch,yaw,quality`: 7 IEEE-754 float32 little-endian |
| 39 | 2 | CRC-16/CCITT-FALSE sur offsets 2..38, little-endian |

Les angles de pose sont en degrés dans `[-180,180]`, qualité dans `[0,1]`, position en
mètres. Le manche RC commande roll/pitch en angles demandés (±30°) et le lacet
en vitesse de rotation (±90°/s), pas en cap absolu, puisque le MPU6050 seul ne
fournit pas de référence de cap. Les voies RC et leur sens dans `rc_input.cpp`
restent à confirmer avec le récepteur et la radiocommande réellement choisis. NaN/Inf, angle/qualité hors plage, version/type/taille inconnus et CRC
incorrect sont rejetés. Le parseur récupère la synchronisation après bruit ou
trame invalide. Le numéro de séquence accepte le retour modulo 16 bits (delta
strictement inférieur à 32768). Le codec correspondant, sans dépendance externe,
est `../Pi/vision_uart.py` et ses tests sont exécutables avec
`python3 ../Pi/test_vision_uart.py`. Le timestamp Pi n'est pas comparé à `millis()`
ESP32. Les trames sont des mesures de vision, **pas des commandes directes de
gouvernes**; aucune fusion externe n'est activée tant que les conventions de
repère/calibration de la caméra ne sont pas définies.

## UART Pi → ESP32 : consigne de guidage version 1

Le même lien UART2 transporte les trames de pose (type 1) et, uniquement si
l'autonomie est explicitement autorisée, des consignes de guidage (type 2).
La trame type 2 fait 31 octets, les champs sont little-endian et les floats
IEEE-754 binary32 : sync `A5 5A`, version `1`, type `2`, taille `24`, séquence
`uint16`, timestamp Pi `uint32 ms`, roll `float` en degrés, pitch `float` en
degrés, taux de lacet `float` en degrés/s, validité `uint16 ms`, qualité de
navigation `float` dans `[0,1]`, puis CRC-16/CCITT-FALSE sur les octets de
version au dernier champ, CRC little-endian. Le détail exécutable est dans
`protocol.h` et `../Pi/vision_uart.py`; les tests Python compilent une fixture
C++ et comparent les octets exacts.

L'ESP32 ne compare pas le timestamp distant à `millis()` : la durée de validité
commence à la réception locale. Les limites du codec sont roll ±25°, pitch ±8°,
taux de lacet ±25°/s, validité 1–500 ms. Une qualité sous le seuil configuré
(0,60 par défaut) est refusée. La sélection AUTO physique n'autorise rien à elle
seule : `AUTO_MODE_ENABLED` reste `false`; une activation future doit exiger un
sélecteur RC AUTO, une action d'armement RC, une commande RC fraîche pour
reprise immédiate, un capteur sain et des consignes reçues régulièrement.
Perte de navigation avec RC valide retombe en stabilisation RC; perte RC ou IMU
entraîne FAILSAFE/défaut capteur. Le failsafe logiciel centre les servos selon
le comportement actuel, ce qui n'est pas une stratégie de retour de gouvernes
validée pour le modèle réel.

## Vision logicielle

Les tests et benchs ArUco synthétiques restent dans `../Sol`. Aucun résultat PC
ne garantit la latence ou le rappel sur Raspberry Pi. Le test e2e logiciel
`test/native/test_protocol_e2e.py` génère un marqueur, le détecte puis exécute
les tests natifs de transport et de modes; aucune sortie moteur réelle n'est
utilisée. Exécuter aussi `python3 ../Pi/test_vision_uart.py` pour le contrat
croisé pose/guidage Python/C++ et `python3 test/native/run_native.py` pour les
modes et limites sans matériel.
