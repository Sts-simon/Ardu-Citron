# Benchmark ArUco

- Date : `2026-09-23 08:02:55`
- Python : `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]`
- OpenCV : `5.0.0`
- NumPy : `2.4.6`
- CPU : `x86_64`
- Threads OpenCV demandés : `4`
- Dataset : `/mnt/1to/ODB/Ardu-Citron-3/Sol/benchmark_dataset`
- Images utilisées : `400`

## Résultats principaux

| variant | benchmark | id_recall | id_precision | id_f1 | wrong_id | fp | fn | mean_ms | p95_ms | p99_ms | fps_eq | aruco_ms | dedup_ms | track_ms | full | roi | retry_ok | rescan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| speed_roi | static | 16.25% | 99.24% | 27.93% | 1 | 0 | 674 | 12.484 | 18.820 | 20.446 | 80.1 | 12.435 | 0.014 | 0.012 | 400 | 0 | 0 | 0 |
| speed_roi | temporal | 29.04% | 100.00% | 45.00% | 0 | 0 | 721 | 8.377 | 17.756 | 21.794 | 119.4 | 8.188 | 0.020 | 0.059 | 385 | 95 | 0 | 11 |
| balanced_roi | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 15.715 | 24.816 | 28.720 | 63.6 | 15.650 | 0.019 | 0.020 | 400 | 0 | 0 | 0 |
| balanced_roi | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 8.729 | 19.006 | 23.259 | 114.6 | 8.475 | 0.024 | 0.090 | 319 | 161 | 0 | 13 |
| quality_roi | static | 37.59% | 98.70% | 54.45% | 4 | 0 | 499 | 22.824 | 35.710 | 41.772 | 43.8 | 22.049 | 0.716 | 0.028 | 400 | 0 | 0 | 0 |
| quality_roi | temporal | 53.64% | 100.00% | 69.83% | 0 | 0 | 471 | 11.641 | 24.998 | 31.132 | 85.9 | 10.582 | 0.821 | 0.094 | 293 | 187 | 0 | 29 |
| speed_full_only | static | 16.25% | 99.24% | 27.93% | 1 | 0 | 674 | 12.847 | 19.316 | 22.042 | 77.8 | 12.801 | 0.013 | 0.012 | 400 | 0 | 0 | 0 |
| speed_full_only | temporal | 27.85% | 100.00% | 43.57% | 0 | 0 | 733 | 8.911 | 15.907 | 20.092 | 112.2 | 8.806 | 0.020 | 0.058 | 480 | 0 | 0 | 0 |
| balanced_full_only | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 17.243 | 28.773 | 43.705 | 58.0 | 17.178 | 0.019 | 0.019 | 400 | 0 | 0 | 0 |
| balanced_full_only | temporal | 44.88% | 100.00% | 61.96% | 0 | 0 | 560 | 11.542 | 20.741 | 24.346 | 86.6 | 11.390 | 0.025 | 0.096 | 480 | 0 | 0 | 0 |
| quality_full_only | static | 37.59% | 98.70% | 54.45% | 4 | 0 | 499 | 21.255 | 31.646 | 34.247 | 47.0 | 20.585 | 0.618 | 0.026 | 400 | 0 | 0 | 0 |
| quality_full_only | temporal | 53.54% | 100.00% | 69.74% | 0 | 0 | 472 | 15.381 | 25.055 | 34.280 | 65.0 | 14.392 | 0.860 | 0.096 | 480 | 0 | 0 | 0 |
| balanced_roi_side12 | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 15.934 | 23.807 | 26.600 | 62.8 | 15.874 | 0.018 | 0.018 | 400 | 0 | 0 | 0 |
| balanced_roi_side12 | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 9.981 | 22.865 | 28.991 | 100.2 | 9.700 | 0.025 | 0.099 | 319 | 161 | 0 | 13 |
| balanced_roi_side16 | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 15.956 | 24.788 | 27.898 | 62.7 | 15.893 | 0.019 | 0.019 | 400 | 0 | 0 | 0 |
| balanced_roi_side16 | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 9.019 | 19.566 | 25.859 | 110.9 | 8.760 | 0.024 | 0.091 | 319 | 161 | 0 | 13 |
| balanced_roi_side20 | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 16.953 | 25.491 | 32.179 | 59.0 | 16.887 | 0.019 | 0.019 | 400 | 0 | 0 | 0 |
| balanced_roi_side20 | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 9.287 | 21.668 | 25.695 | 107.7 | 9.030 | 0.025 | 0.090 | 319 | 161 | 0 | 13 |
| balanced_roi_full8 | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 19.825 | 36.166 | 52.715 | 50.4 | 19.748 | 0.021 | 0.022 | 400 | 0 | 0 | 0 |
| balanced_roi_full8 | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 9.841 | 22.094 | 31.376 | 101.6 | 9.547 | 0.026 | 0.108 | 319 | 161 | 0 | 13 |
| balanced_roi_full16 | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 16.242 | 25.214 | 29.217 | 61.6 | 16.178 | 0.019 | 0.019 | 400 | 0 | 0 | 0 |
| balanced_roi_full16 | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 8.649 | 18.462 | 21.924 | 115.6 | 8.409 | 0.022 | 0.084 | 319 | 161 | 0 | 13 |
| balanced_roi_full32 | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 16.073 | 24.949 | 28.265 | 62.2 | 16.009 | 0.019 | 0.019 | 400 | 0 | 0 | 0 |
| balanced_roi_full32 | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 8.658 | 18.788 | 23.580 | 115.5 | 8.428 | 0.021 | 0.082 | 319 | 161 | 0 | 13 |
| balanced_roi_thr1 | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 24.406 | 38.114 | 49.736 | 41.0 | 24.358 | 0.013 | 0.013 | 400 | 0 | 0 | 0 |
| balanced_roi_thr1 | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 10.986 | 28.925 | 36.065 | 91.0 | 10.821 | 0.015 | 0.057 | 319 | 161 | 0 | 13 |
| balanced_roi_thr2 | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 16.225 | 28.649 | 46.229 | 61.6 | 16.170 | 0.015 | 0.015 | 400 | 0 | 0 | 0 |
| balanced_roi_thr2 | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 12.060 | 34.282 | 63.820 | 82.9 | 11.800 | 0.023 | 0.087 | 319 | 161 | 0 | 13 |
| balanced_roi_thr3 | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 15.952 | 24.435 | 28.463 | 62.7 | 15.892 | 0.018 | 0.017 | 400 | 0 | 0 | 0 |
| balanced_roi_thr3 | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 8.875 | 18.100 | 23.613 | 112.7 | 8.626 | 0.023 | 0.089 | 319 | 161 | 0 | 13 |
| balanced_roi_thr4 | static | 27.79% | 100.00% | 43.50% | 0 | 0 | 582 | 17.975 | 28.881 | 31.280 | 55.6 | 17.907 | 0.020 | 0.020 | 400 | 0 | 0 | 0 |
| balanced_roi_thr4 | temporal | 45.18% | 100.00% | 62.24% | 0 | 0 | 557 | 9.390 | 19.941 | 23.338 | 106.5 | 9.094 | 0.025 | 0.098 | 319 | 161 | 0 | 13 |

## Lecture des métriques

- `id_recall` : proportion des vrais marqueurs retrouvés avec le bon ID.
- `id_precision` : proportion des prédictions correspondant au bon ID.
- `localization_recall` : marqueur correctement localisé même si l'ID est faux.
- `wrong_id` : bonne localisation mais mauvais ID valide.
- `mean/p95/p99` : latence de `det.detect()` uniquement, hors lecture PNG.
- `aruco/dedup/track` : détail interne chronométré par la v2.1.
- `fps_eq` : 1000 / latence moyenne ; ce n'est pas un FPS caméra garanti.

## Attention benchmark temporel

Le benchmark temporel réutilise les scènes générées mais applique des petites transformations géométriques déterministes pour solliciter le tracking et les ROI. Il est utile pour comparer les stratégies ROI, mais ne remplace pas une vraie séquence vidéo issue du vol.
