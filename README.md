<div align="center">

# 🍋 Ardu-Citron

### Prototype expérimental de contrôle et de guidage pour planeur RC

Firmware ESP32 · perception et guidage Python · émulation logicielle 6-DOF

![Status](https://img.shields.io/badge/status-expérimental-orange)
![Platform](https://img.shields.io/badge/ESP32-Arduino%20%2F%20PlatformIO-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

**Banc d’apprentissage et de prototypage, pas un autopilote certifié. L’Auto matériel est désactivé par défaut.**

</div>

---

## État actuel

- **ESP32** : lecture MPU6050, calibration et surveillance IMU, Kalman, PID, mixage et
  limitation des sorties servo, entrées RC PWM, modes Manuel/Stabilisé/Auto, failsafe et UART2.
- **Raspberry Pi / Python** : détection ArUco/PnP, estimation de navigation expérimentale,
  guidage géorepéré, runtime caméra et codec UART compatible avec le firmware.
- **Émulation** : modèle d’avion logiciel 6-DOF, caméra et capteurs synthétiques, détection
  ArUco, contrôleur C++ natif, protocole UART et journalisation.
- **Interface de simulation** : interface de bureau Tkinter avec paramètres classés par
  catégories, lancement de la simulation et aperçu des résultats.
- **Outils de vision** : détection, tests et benchmarks ArUco dans `Ardu-Citron-3/Sol/`.

Les tests logiciels ne valident ni le câblage ni le comportement en vol.

## Organisation

```text
Ardu-Citron/
├── README.md
├── requirements.txt                 Dépendances d’outils à la racine
└── Ardu-Citron-3/
    ├── Code/
    │   ├── ESP_32/
    │   │   ├── main/                 Firmware et modules C++
    │   │   ├── test/native/          Tests natifs et contrat e2e logiciel
    │   │   ├── Actionneur_Check/     Croquis de banc actionneurs
    │   │   ├── Teste_Mpu/            Croquis de banc MPU6050
    │   │   ├── platformio.ini
    │   │   └── README.md             Build, sécurité, brochage et UART
    │   ├── Pi/                      Codec UART, navigation, runtime caméra, tests
    │   ├── Emulation/               Simulateur 6-DOF, UI Tkinter, tests, config
    │   └── Electronique/Pcb/        Documents de conception PCB
    ├── Model_3d/                    Modèles mécaniques
    └── Sol/                         Détection ArUco, génération et benchmarks
```

## Émulation et interface de bureau

Prérequis : Python 3.10+, `g++`, OpenCV contrib, NumPy et Matplotlib.

```bash
cd Ardu-Citron-3/Code/Emulation
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Lancer directement le simulateur :

```bash
python simulator.py --duration 5 --out output-smoke
# ou : python simulator.py
```

Ouvrir l’interface native :

```bash
python sim_ui.py
```

L’interface regroupe les paramètres en onglets (simulation, avion, caméra, capteurs,
navigation, guidage et scénarios RC). Les vecteurs, marqueurs et scénarios restent éditables
dans des champs JSON. Les résultats et la configuration exacte de chaque exécution sont
rangés dans `output-ui/run-<date>/` : journal `flight_log.csv`, résumé `summary.json` et
courbe `flight_summary.png`.

Tkinter doit être installé avec Python et un environnement graphique doit être disponible.
Sur les distributions minimales Debian/Ubuntu, installer `python3-tk`; la ligne de commande
`python simulator.py` n’a pas besoin de Tkinter. L’Auto est désactivé dans la configuration
par défaut. La simulation ne pilote aucun matériel.

## Tests logiciels

Depuis `Ardu-Citron-3/Code/Emulation` :

```bash
python -m unittest test_simulator.py
```

Depuis `Ardu-Citron-3/Code/ESP_32` :

```bash
python3 test/native/run_native.py
python3 ../Pi/test_vision_uart.py
python3 ../Pi/test_autonomy.py
python3 test/native/test_protocol_e2e.py
```

Ces tests couvrent des modules natifs, les contrats UART Python/C++, des cas de guidage et une
chaîne ArUco synthétique. Ils ne remplacent pas le build ESP32 ni les essais sur banc.

## Firmware ESP32

La configuration PlatformIO cible `esp32dev`, Arduino et `ESP32Servo` (versions dans
`platformio.ini`). Depuis `Ardu-Citron-3/Code/ESP_32` :

```bash
python3 -m pip install -r requirements-dev.txt
pio run -e esp32
```

Confirmer le modèle précis de la carte avant flash. Le câblage prévu inclut le MPU6050 sur
I²C GPIO22/23, les servos sur GPIO26/27/14, et l’UART2 Pi sur RX GPIO16 / TX GPIO17,
115200 bauds. Voir [la documentation ESP32](Ardu-Citron-3/Code/ESP_32/README.md) et
[le brochage](Ardu-Citron-3/Code/ESP_32/pinout.txt) ; vérifier les broches et niveaux logiques
sur le matériel réellement utilisé.

### Modes et sécurité

Le démarrage est conditionné par la calibration et une première lecture IMU valide. L’armement
requiert des entrées RC récentes. La voie de mode sélectionne le manuel direct, la stabilisation
RC ou une demande Auto. Les consignes Pi→ESP32 sont bornées, versionnées et protégées par CRC.
La fraîcheur est vérifiée avec l’horloge locale ESP32.

**`AUTO_MODE_ENABLED` reste `false` dans `main/config.h`. Ne pas l’activer pour voler.**
Les gains PID, sens et limites servo, voies RC, stratégie de gouvernes en failsafe et paramètres
aérodynamiques doivent être validés pour l’aéronef. Un test logiciel réussi ne prouve pas que
la réponse matérielle est sûre.

Avant toute expérimentation en vol, faire une revue électrique, confirmer les manches et les
sens de gouvernes sur banc avec hélice/moteur désactivé, et tester les pertes RC/UART/IMU.
Suivre les pratiques de sécurité aéronautique et demander une revue à un pilote expérimenté.

## Raspberry Pi et UART

Depuis `Ardu-Citron-3/Code/Pi` :

```bash
python3 -m pip install -r requirements.txt
python3 autonomy_runtime.py --dry-run
```

Le mode `--dry-run` lit une caméra accessible sans ouvrir le port UART. Le runtime ne dispose
pas encore d’adaptateurs matériels prêts à l’emploi pour `imu_provider` et `gps_provider` : une
navigation valide exige de fournir et valider ceux adaptés au matériel. Les paramètres et
coordonnées des repères dans `autonomy.json` sont des exemples, pas une calibration universelle.

Le protocole v1 utilise deux types de message : pose (type 1) et consigne de guidage (type 2),
avec sync `A5 5A`, champs little-endian et CRC-16/CCITT-FALSE. Pour le lien : TX Pi → RX ESP32
GPIO16, RX Pi ← TX ESP32 GPIO17, masse commune, niveaux logiques 3,3 V. Ne pas appliquer de
signal 5 V à un GPIO. Vérifier le brochage du Pi, de l’ESP32 et de tout convertisseur avant
connexion.

## Limites connues

- Les paramètres de l’avion et l’aérodynamique de l’émulateur sont provisoires; la poussée est
  nulle et le modèle n’est pas CFD ni certifié.
- La vision, la navigation complémentaire, la qualité des mesures et la calibration caméra
  doivent être validées avec les capteurs et repères réels.
- L’émulateur ne représente pas les interruptions, délais électriques, vibrations, alimentation,
  actionneurs réels ni toutes les dynamiques de vol.
- Le runtime Pi nécessite encore les adaptateurs matériels IMU/GPS.
- Les outils de données et benchmarks dans `Sol/` sont des outils de recherche, séparés du
  runtime de vol.

Conserver `AUTO_MODE_ENABLED=false` tant que l’intégration au sol et la stratégie de sécurité
propre à la cellule ne sont pas validées.
