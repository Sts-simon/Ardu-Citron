use std::path::Path;

// =========================================================================
// 1. CONFIGURATION PHYSIQUE & TECHNIQUE (Identique à generate_dataset_v2.py)
// =========================================================================
const IMU_ALPHA: f32 = 0.98;         // 98% Gyro, 2% Accéléro (Filtre complémentaire)
const DT: f32 = 1.0 / 500.0;         // Pas de temps entre deux images (0.002s)
const FRAMES_PER_TRAJECTORY: usize = 500; // Découpage des blocs de trajectoires
const NUM_TRAJECTORIES_TO_SIMULATE: usize = 3; // Exemple sur 3 trajectoires consécutives

// =========================================================================
// 2. MODULE : FILTRE DE KALMAN LINEAIRE 3D (X, Y, Z)
// =========================================================================
#[derive(Debug, Clone)]
struct KalmanFilter3D {
    x: [f32; 6],  // État cinématique : [x, vx, y, vy, z, vz]
    p: [f32; 6],  // Variance de l'erreur d'estimation (Diagonale simplifiée)
    q: f32,       // Bruit thermique / Processus (dynamique de vol)
    r: f32,       // Bruit de mesure (variance native moyenne du CNN ~40cm)
}

impl KalmanFilter3D {
    /// Initialise le filtre avec la première position brute vue par le CNN
    fn new(init_x: f32, init_y: f32, init_z: f32) -> Self {
        Self {
            x: [init_x, 0.0, init_y, 0.0, init_z, 0.0],
            p: [1.0, 10.0, 1.0, 10.0, 1.0, 10.0], // Incertitude initiale (vitesse inconnue)
            q: 0.1,  // Souplesse face aux accélérations réelles du drone
            r: 0.16, // Variance de l'erreur du CNN (0.40m * 0.40m)
        }
    }

    /// Étape de Prédiction (Lois cinématiques : Position = Vitesse * dt)
    fn predict(&mut self, dt: f32) {
        self.x[0] += self.x[1] * dt; // X
        self.x[2] += self.x[3] * dt; // Y
        self.x[4] += self.x[5] * dt; // Z

        // Évolution des incertitudes associées
        for i in (0..6).step_by(2) {
            self.p[i] += self.p[i+1] * dt * 2.0 + self.q * dt;
            self.p[i+1] += self.q * dt;
        }
    }

    /// Étape de Correction (Fusion mathématique avec la mesure brute dé-normalisée du CNN)
    fn update(&mut self, z_x: f32, z_y: f32, z_z: f32) {
        let measurements = [z_x, z_y, z_z];
        for (idx, &z_meas) in measurements.iter().enumerate() {
            let state_idx = idx * 2;
            let innovation = z_meas - self.x[state_idx];
            let k_gain = self.p[state_idx] / (self.p[state_idx] + self.r);
            
            self.x[state_idx] += k_gain * innovation;               // Correction Position
            self.x[state_idx + 1] += (k_gain / DT) * innovation;    // Estimation de la Vitesse réinjectée
            self.p[state_idx] *= 1.0 - k_gain;                      // Réduction de l'incertitude
        }
    }
}

// =========================================================================
// 3. MODULE : FILTRE COMPLÉMENTAIRE ATTITUDE (MPU6050)
// =========================================================================
struct Mpu6050Filter {
    roll_imu: f32,
    pitch_imu: f32,
    yaw_imu: f32,
}

impl Mpu6050Filter {
    /// Initialise l'IMU avec une assiette de départ (angles d'origine)
    fn new(init_roll: f32, init_pitch: f32, init_yaw: f32) -> Self {
        Self {
            roll_imu: init_roll,
            pitch_imu: init_pitch,
            yaw_imu: init_yaw,
        }
    }

    /// Applique l'intégration Gyro + recalage Accéléro identique à simulate_mpu6050_imu
    fn process_imu_data(&mut self, gyro_dps: [f32; 3], accel_g: [f32; 3], dt: f32) {
        // 1. Intégration brute du Gyroscope (Angles prédits)
        let roll_pred  = self.roll_imu  + gyro_dps[0] * dt;
        let pitch_pred = self.pitch_imu + gyro_dps[1] * dt;
        self.yaw_imu   += gyro_dps[2] * dt; // Le Yaw dérive uniquement au gyro (pas de magnétomètre)

        // 2. Angles géométriques déduits des accéléromètres (Trigonométrie inverse sur la gravité)
        let roll_accel  = accel_g[1].asin().to_degrees();
        let pitch_accel = (-accel_g[0]).asin().to_degrees();

        // 3. Fusion Complémentaire à mémoire longue (98% Gyro haute freq / 2% Accel stable)
        self.roll_imu  = IMU_ALPHA * roll_pred  + (1.0 - IMU_ALPHA) * roll_accel;
        self.pitch_imu = IMU_ALPHA * pitch_pred + (1.0 - IMU_ALPHA) * pitch_accel;
    }
}

// =========================================================================
// 4. BOUCLE PRINCIPALE ET ANALYSE CHRONOLOGIQUE PAR TRAJECTOIRE
// =========================================================================
fn main() {
    println!("========================================================================");
    println!("📡 [Ardu-Citron] SYSTEME AVIONIQUE EMBARQUE - EXECUTION PAR TRAJECTOIRES");
    println!("========================================================================");

    // Vérification de sécurité pour le fichier ONNX de production
    let model_path = Path::new("tiny_drone_localizer.onnx");
    if model_path.exists() {
        println!("✅ Modèle de production 'tiny_drone_localizer.onnx' prêt.");
    } else {
        println!("⚠️ Mode Simulation active (Le fichier ONNX n'est pas à la racine du projet Rust).");
    }
    println!("⏱️ Fréquence capteurs : 500 Hz | Fenêtrage : Continuité par lots de {} images\n", FRAMES_PER_TRAJECTORY);

    // Simulation de plusieurs trajectoires distinctes (de 500 en 500 images)
    for traj_idx in 1..=NUM_TRAJECTORIES_TO_SIMULATE {
        println!("------------------------------------------------------------------------");
        println!("🚀 DEBUT DE LA TRAJECTOIRE N°{} (Frames : {} à {})", 
                 traj_idx, (traj_idx-1)*FRAMES_PER_TRAJECTORY, (traj_idx*FRAMES_PER_TRAJECTORY)-1);
        println!("------------------------------------------------------------------------");

        // --- INSTANCIATION / REINITIALISATION DES FILTRES ---
        // Très important : Chaque début de trajectoire correspond à une nouvelle phase ou un nouveau marqueur.
        // Les filtres doivent effacer l'historique cinématique précédent pour coller à la réalité physique.
        let mut kalman: Option<KalmanFilter3D> = None;
        let mut imu_filter = Mpu6050Filter::new(0.0, 0.0, 0.0);

        // Analyse temporelle pas à pas des 500 images de la trajectoire courante
        for frame_idx in 0..FRAMES_PER_TRAJECTORY {
            
            // --- 1. SIMULATION ET RECEPTION DES DONNEES SYNCHRONISEES (I2C + VISION) ---
            // Simule l'évolution des signaux bruts générés par generate_dataset_v2.py
            let cnn_output_raw = generate_simulated_cnn_tensors(frame_idx);
            let (gyro_raw, accel_raw) = generate_simulated_imu_signals(frame_idx);

            // --- 2. TRAITEMENT DE L'ATTITUDE HAUTE FRÉQUENCE (IMU) ---
            imu_filter.process_imu_data(gyro_raw, accel_raw, DT);

            // --- 3. DE-NORMALISATION MATHÉMATIQUE DES UNITÉS DU CNN ---
            // Formules inverses basées sur la configuration et l'apprentissage du modèle
            let cnn_x = cnn_output_raw[0] * 3.0; // Borné à [-3m, 3m]
            let cnn_y = cnn_output_raw[1] * 3.0; // Borné à [-3m, 3m]
            let cnn_z = (cnn_output_raw[2] * (6.0 - 2.0)) + 2.0; // Plage d'altitude [2m, 6m]

            let cnn_roll  = cnn_output_raw[3] * 35.0; // Max 35°
            let cnn_pitch = cnn_output_raw[4] * 20.0; // Max 20°
            let cnn_yaw   = cnn_output_raw[5] * 45.0; // Max 45°

            // --- 4. GESTION ET ALIMENTATION DU FILTRE DE KALMAN (POSITION) ---
            if kalman.is_none() {
                // Première frame du bloc de 500 : On accroche (lock) le filtre sur la position initiale calculée
                kalman = Some(KalmanFilter3D::new(cnn_x, cnn_y, cnn_z));
            } else if let Some(ref mut k_filter) = kalman {
                // Frames suivantes : Étape classique Prédiction -> Correction
                k_filter.predict(DT);
                k_filter.update(cnn_x, cnn_y, cnn_z);
            }

            // --- 5. LOG ET CONTRÔLE DES JALONS DE PROGRESSION (Toutes les 250 frames) ---
            if frame_idx == 0 || frame_idx == 250 || frame_idx == 499 {
                if let Some(ref k) = kalman {
                    println!("📍 Frame #{:<3} | T: {:.3}s", frame_idx, (frame_idx as f32) * DT);
                    println!("   ↳ CNN Brut  -> Pos [X: {:>5.2}m, Y: {:>5.2}m, Z: {:>5.2}m] | Ang [R: {:>5.1}°, P: {:>5.1}°, Y: {:>5.1}°]", 
                             cnn_x, cnn_y, cnn_z, cnn_roll, cnn_pitch, cnn_yaw);
                    println!("   ↳ KALMAN Est-> Pos [X: {:>5.2}m, Y: {:>5.2}m, Z: {:>5.2}m] | Vit [Vx: {:>5.2}m/s, Vy: {:>5.2}m/s, Vz: {:>5.2}m/s]", 
                             k.x[0], k.x[2], k.x[4], k.x[1], k.x[3], k.x[5]);
                    println!("   ↳ IMU Fusion-> Ang [Roll: {:>5.1}° | Pitch: {:>5.1}° | Yaw: {:>5.1}°]", 
                             imu_filter.roll_imu, imu_filter.pitch_imu, imu_filter.yaw_imu);
                    println!("   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
                }
            }
        }
        println!("✅ Trajectoire N°{} terminée avec succès et purgée.\n", traj_idx);
    }
}

// =========================================================================
// GENERATEURS DE FLUX POUR EMULER LA LECTURE DES FICHIERS JSON EN RUST
// =========================================================================

/// Émule la sortie brute du tenseur ONNX (6 float normalisés) au cours du temps
fn generate_simulated_cnn_tensors(frame: usize) -> [f32; 6] {
    let phase = (frame as f32) * DT * 2.0 * std::f32::consts::PI;
    [
        (phase.cos() * 0.4) + 0.05, // X_norm
        (phase.sin() * 0.3) - 0.02, // Y_norm
        0.5 + (frame as f32 * 0.0005), // Z_norm (le drone monte doucement de 4m à 5m)
        (phase.cos() * 0.2),        // Roll_norm
        (phase.sin() * 0.15),       // Pitch_norm
        (frame as f32 * 0.001)      // Yaw_norm (virage constant de lacet)
    ]
}

/// Émule les registres bruts lus sur le bus I2C du MPU6050 (Gyro en dps, Accel en G)
fn generate_simulated_imu_signals(frame: usize) -> ([f32; 3], [f32; 3]) {
    let phase = (frame as f32) * DT * 2.0 * std::f32::consts::PI;
    
    // Gyroscope (°/s) avec un léger jitter de vibration moteur superposé
    let gyro_x = phase.cos() * 15.0 + 0.5; 
    let gyro_y = phase.sin() * 8.0 - 0.2;
    let gyro_z = 5.0; // vitesse constante sur le lacet
    
    // Accéléromètre (G) convertissant la gravité terrestre
    let accel_x = -(phase.sin() * 5.0).to_radians().sin();
    let accel_y = (phase.cos() * 10.0).to_radians().sin();
    let accel_z = 0.98; // Équilibre vertical proche de 1G

    ([gyro_x, gyro_y, gyro_z], [accel_x, accel_y, accel_z])
}
