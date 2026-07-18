# Contrôleur de vol ESP32 — Planeur RC (maintien d'attitude)

## Dépendance à installer

Ce projet utilise la bibliothèque **ESP32Servo** (par Kevin Harrington), à
installer via le gestionnaire de bibliothèques de l'Arduino IDE
(*Outils > Gérer les bibliothèques… > rechercher "ESP32Servo"*).

Tout le reste (Wire, I2C) fait partie du core Arduino-ESP32.

## Structure

```
main/
├── main.ino       -> setup() / loop(), scheduler multi-fréquence, non bloquant
├── config.h       -> TOUTE la configuration (brochage, consignes, gains, fréquences...)
├── imu.h/.cpp     -> pilote MPU6050 (registre direct, sans lib externe), calibration
├── kalman.h/.cpp  -> filtre de Kalman 2 états (angle/biais), un par axe
├── pid.h/.cpp     -> PID générique (anti-windup, dérivée filtrée, sortie limitée)
├── mixer.h/.cpp   -> mixage Roll/Pitch/Yaw -> 3 servos, plusieurs presets
├── failsafe.h/.cpp-> architecture de sécurité, prête, RC pas encore câblé
└── buzzer.h/.cpp  -> bips non bloquants (tone()/noTone())
```

## Pour régler le vol

Tout se passe dans `config.h` :
- **Consignes** : `SETPOINT_ROLL_DEG`, `SETPOINT_PITCH_DEG`, `SETPOINT_YAW_DEG`
- **Gains PID** : `Kp_roll` / `Ki_roll` / `Kd_roll`, etc.
- **Mixage** : `ACTIVE_MIXING_PRESET` (`STANDARD_AIL_ELEV_RUD`, `VTAIL`, `ELEVON`)
- **Sens/trim servo** : `SERVO_DIR_x`, `SERVO_TRIM_x_DEG`
- **Fréquences** : `FREQ_IMU_HZ`, `FREQ_PID_HZ`, `FREQ_SERVO_HZ`, etc.

