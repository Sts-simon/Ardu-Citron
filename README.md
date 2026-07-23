# 🍋 Ardu-Citron

Ardu-Citron est un projet personnel de développement d'un drone autonome miniature.
L'objectif n'est pas de créer un simple appareil volant, mais d'explorer et de comprendre les différentes technologies qui permettent un système autonome : électronique embarquée, contrôle de vol, vision par ordinateur, localisation et intelligence artificielle.

Le projet est développé de manière expérimentale, avec une volonté de comprendre chaque couche du système, du capteur jusqu'aux algorithmes de décision.

---

## 🚁 Présentation

Ardu-Citron est un drone à voilure fixe conçu pour évoluer dans un environnement d'abord intérieur.

Contrairement aux drones multirotors classiques, le projet utilise une architecture avion :

- aile fixe pour améliorer l'efficacité énergétique ;
- stabilisation embarquée ;
- caméra pour la perception de l'environnement ;
- localisation par vision.

L'objectif final est d'obtenir un système capable de voler de manière autonome tout en restant suffisamment simple pour que chaque élément puisse être étudié et amélioré.

---

# 🎯 Objectifs du projet

Les principaux objectifs sont :

- Comprendre la stabilisation d'un aéronef.
- Développer un autopilote embarqué.
- Explorer la fusion de capteurs.
- Utiliser la vision artificielle pour la localisation via un CNN.
- Expérimenter avec des modèles d'intelligence artificielle embarqués.
- Optimiser les parties critiques en performances.

Ardu-Citron est avant tout un laboratoire expérimental permettant d'apprendre par la conception et l'expérimentation.

---

# 🏗️ Architecture générale

Le système est séparé en plusieurs parties.
                Caméra
                   |
                   v
          Raspberry Pi Zero 2 W
                   |
    Vision / IA / Localisation / Navigation
                   |
                   |
            Communication
                   |
                   v
                ESP32
                   |
    Stabilisation / Capteurs / Actionneurs
                   |
          Servos + Moteur

---

# 💻 Technologies utilisées

## ESP32 - Contrôle temps réel

Langage : C++

L'ESP32 est responsable des tâches nécessitant une réponse rapide :

- lecture des capteurs ;
- boucle de stabilisation ;
- contrôle des servomoteurs ;
- commande moteur.

La boucle de contrôle vise une fréquence élevée afin d'assurer une stabilisation réactive.

---

## Raspberry Pi - Intelligence embarquée

Langages : Rust

Le Raspberry Pi réalise les tâches plus complexes :

- traitement d'image ;
- détection de marqueurs ArUco ;
- localisation ;
- calculs de navigation ;
- exécution de modèles IA.

Python est utilisé pour le développement rapide et l'expérimentation.

Rust est utilisé pour les modules où les performances sont importantes, notamment certains traitements temps réel.

---

# 👁️ Vision et localisation

Le projet utilise la vision par ordinateur afin de permettre au drone de comprendre son environnement.

Les recherches actuelles portent notamment sur :

- détection de marqueurs ArUco ;
- estimation de position ;
- fusion vision + IMU ;
- accélération des calculs.

Un système de localisation basé sur caméra permet d'éviter de dépendre uniquement du GPS, inutilisable dans un gymnase.


---

# 🔬 Approche de développement

Le projet suit une approche progressive :

1. Tester les composants individuellement.
2. Comprendre les limitations du matériel.
3. Développer les algorithmes.
4. Mesurer les performances.
5. Optimiser les parties critiques.

Chaque module est développé avec l'objectif de comprendre son fonctionnement plutôt que simplement utiliser une solution existante.

---

# 📊 Performances actuelles

Quelques résultats obtenus pendant les expérimentations :

- Détection et traitement vision optimisés en Rust.
- Benchmarks dépassant plusieurs centaines d'images par seconde sur certains modules.
- Fusion vision + IMU testée.
- Génération et utilisation de datasets pour modèles IA.

Ces résultats sont issus de tests en environnement contrôlé et continuent d'être améliorés.

---

# 🛠️ Installation

## Prérequis

- Python 3.12
- Rust
- PlatformIO ou Arduino IDE
- OpenCV
- PyTorch (pour les modules IA)

Installation Python :

```bash
pip install -r requirements.txt
