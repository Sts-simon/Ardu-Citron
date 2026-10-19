# Pipeline automatise pour les episodes Manim

Ce paquet automatise le flux suivant :

1. lire le texte parle depuis `script_annote.txt` (les timestamps/notes entre parentheses sont ignores ; le texte en gras est conserve) ;
2. lancer une transcription de controle Whisper (`base.en`, optionnelle) ;
3. faire un **alignement force** du texte exact sur la voix avec WhisperX ;
4. convertir les ancres audio en valeurs pour les 58 cles du dictionnaire `C` ;
5. lancer la scene Manim `Unguided` avec ces valeurs ;
6. fusionner le MP4 Manim avec le fichier audio original via FFmpeg. La voix n'est ni decoupee ni ralentie.

Le `cue_map.json` fourni est preconfigure pour le script et l'animation de l'episode 1. Pour un nouvel episode, son mapping doit pointer vers les mots/phrases qui doivent declencher chaque element visuel. Une fois le mapping d'un modele d'animation fait, il est reutilisable tant que les ancres existent dans le nouveau texte.

## 1. Installer les dependances

Dans l'environnement Python ou Manim fonctionne deja, installe WhisperX :

```bash
pip install whisperx
```

Si tu pars d'un environnement neuf sous Linux Mint :

```bash
sudo apt update
sudo apt install ffmpeg python3-venv python3-pip build-essential pkg-config libcairo2-dev libpango1.0-dev python3-dev
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install manim whisperx
```

Les modeles WhisperX sont telecharges lors de leur premier usage. Ils sont ensuite mis en cache localement.

## 2. Tester uniquement l'alignement

Depuis le dossier qui contient les fichiers :

```bash
python serie_pipeline.py script_annote.txt voix_off.mp3 guidage_ep01_court.py \
  --cue-map cue_map.json \
  --scene Unguided \
  --output-dir build_ep01 \
  --align-only
```

Examine `build_ep01/alignment_report.txt`, `build_ep01/word_times.csv` et `build_ep01/whisper_transcription.txt`. Le controle ASR peut signaler une difference de formulation, notamment sur CCIP/CCRP. Le programme s'arrete si une ancre du mapping est introuvable, plutot que de produire une video dont les timings seraient devines.

## 3. Generer toute la video

Quand l'alignement est correct, relance la commande sans `--align-only` :

```bash
python serie_pipeline.py script_annote.txt voix_off.mp3 guidage_ep01_court.py \
  --cue-map cue_map.json \
  --scene Unguided \
  --output-dir build_ep01
```

Pour sauter la transcription de controle et ne faire que l'alignement force (plus rapide, moins de diagnostic) : ajoute `--skip-asr-check`.

## 4. Fichiers crees

- `episode_final.mp4` : animation avec la voix off originale ;
- `animation_silent.mp4` : rendu Manim sans audio, utile pour diagnostiquer ;
- `cues_generated.json` : dictionnaire `C` rempli automatiquement + `END_T` ;
- `animation_with_cues.py` : copie de la scene, capable de charger le JSON genere ;
- `word_times.csv` : timestamps et scores par mot ;
- `alignment_report.txt` : resultats, erreurs et valeurs du dictionnaire ;
- `whisper_transcription.txt` : transcription de controle Whisper, si active ;
- `script_clean.txt` : texte parle transmis a l'aligner.

## 5. Adapter `cue_map.json`

Chaque cle doit correspondre a une cle existante du dictionnaire `C` dans le fichier Manim. Le mapping fourni utilise deux types d'ancres.

**Option recommandee : index d'un segment en gras.** Cela survit aux changements de formulation, a condition de conserver l'ordre des segments en gras dans le script. Les index commencent a zero.

```json
"ccip": {
  "bold_index": 17,
  "word_index": 0,
  "edge": "start",
  "offset": 0.0
}
```

Dans le script fourni, le segment en gras numero 17 est `CCIP`. Le programme l'aligne sur sa vraie position dans l'audio. `word_index` permet de viser un mot a l'interieur d'un segment en gras de plusieurs mots.

**Autre option : une phrase de reference.** Utile lorsqu'une animation doit apparaitre sur un mot qui n'est pas en gras, par exemple une transition au debut d'une phrase.

```json
"to_air": {
  "phrase": "So the intelligence must be in the aircraft",
  "word_index": 0,
  "edge": "start",
  "offset": 0.0
}
```

- `bold_index` : index du segment en gras, en commencant a zero ;
- `phrase` : une phrase unique ou assez distinctive du texte prononce, pour une ancre non en gras ;
- `word_index` : index du mot a l'interieur du segment en gras ou de `phrase`, en commencant a zero ;
- `edge` : `start` pour le debut du mot, `end` pour sa fin ;
- `offset` : decalage en secondes, positif ou negatif ;
- `occurrence` : occurrence a utiliser si `phrase` apparait plusieurs fois, en commencant a 1.

Les offsets sont des choix visuels. Par exemple, `offset: -0.15` fait apparaitre un element 150 ms avant le mot ancre. Si l'ordre chronologique de deux cues s'inverse, le programme le signale dans le rapport ; Manim peut alors declencher le deuxieme cue en retard.

Le script doit contenir le texte prononce dans le bon ordre. Les marques `**mot**`, les marqueurs `**[6.0]**` et les notes de temps `*(6.2)*` sont supprimes avant l'alignement ; le gras ne change donc pas ce que WhisperX aligne.

## Limites a savoir

- L'alignement force suppose que la voix dit bien le texte fourni. La transcription de controle sert a signaler les ecarts, mais ne remplace pas une verification auditive.
- Les acronymes techniques peuvent etre mal reconnus par Whisper ; les temps sont alignes sur le texte de reference, et le rapport aide a reperer les anomalies.
- Manim ne peut pas executer deux animations longues en meme temps simplement parce que leurs cues sont proches. Si le rendu signale des cues en retard, il faudra raccourcir certaines durees `run_time` ou ajuster les offsets dans `cue_map.json`.
- Cette execution assemble l'audio original sans le decouper. Si la video depasse l'audio, FFmpeg ajoute du silence a la fin pour conserver la duree complete de l'animation.
