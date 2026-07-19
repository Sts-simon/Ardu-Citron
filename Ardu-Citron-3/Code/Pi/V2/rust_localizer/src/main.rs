use ort::session::Session;
use ort::value::Value;
use ndarray::Array4;
use std::error::Error;
use std::time::Instant;
use std::fs::File;
use std::io::Read;
use walkdir::WalkDir;
use image::GenericImageView;
use serde::Deserialize;

#[derive(Deserialize)]
struct GroundTruthData {
    marker_id: i32,
    distance_m: f32,
    roll_deg: f32,
    pitch_deg: f32,
    yaw_deg: f32,
    aruco_corners: Vec<Vec<f32>>,
}

struct KalmanFilter1D {
    x: f32,       
    v: f32,       
    p_xx: f32,    
    p_xv: f32,    
    p_vv: f32,    
    q_pos: f32,   
    q_vel: f32,
    r_measure: f32, 
}

impl KalmanFilter1D {
    fn new(initial_pos: f32, cnn_error: f32) -> Self {
        Self {
            x: initial_pos,
            v: 0.0,
            p_xx: 1.0, 
            p_xv: 0.0,
            p_vv: 1.0,
            q_pos: 0.01,  
            q_vel: 0.1,   
            r_measure: cnn_error.powi(2), 
        }
    }

    // ✨ Fonction pour réinitialiser proprement le filtre lors d'un saut de trajectoire
    fn reset(&mut self, current_pos: f32) {
        self.x = current_pos;
        self.v = 0.0;
        self.p_xx = 1.0;
        self.p_xv = 0.0;
        self.p_vv = 1.0;
    }

    fn predict(&mut self, dt: f32) {
        self.x += self.v * dt;
        self.p_xx += dt * (2.0 * self.p_xv + dt * self.p_vv) + self.q_pos;
        self.p_xv += dt * self.p_vv;
        self.p_vv += self.q_vel;
    }

    fn update(&mut self, z_measure: f32) {
        let y = z_measure - self.x;
        let s = self.p_xx + self.r_measure;
        let k_x = self.p_xx / s;
        let k_v = self.p_xv / s;
        
        self.x += k_x * y;
        self.v += k_v * y;
        
        self.p_xx *= 1.0 - k_x;
        self.p_xv *= 1.0 - k_x;
        self.p_vv -= k_v * self.p_xv;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== 🛰️ LOCALISEUR FULL-PIPELINE (AUTO-RESET PAR TRAJECTOIRE) ===");

    let model_path = "tiny_drone_localizer.onnx";
    let dataset_path = "/home/sts33/2eme/ODB/Ardu-Citron-3/Sol/Markers_5/Dataset"; 

    let mut session = Session::builder()?
        .with_intra_threads(2)? 
        .commit_from_file(model_path)?;
    println!("✅ Modèle ONNX chargé avec succès.");

    let dt = 1.0 / 500.0; 
    let cnn_expected_error = 0.49; 
    let mut kalman_z = KalmanFilter1D::new(3.0, cnn_expected_error);

    println!("📂 Scan du dossier '{}' en cours...", dataset_path);
    let mut image_paths = Vec::new();
    for entry in WalkDir::new(dataset_path).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "png" || ext == "jpg" {
                    image_paths.push(path.to_path_buf());
                }
            }
        }
    }
    image_paths.sort(); 

    let total_images = image_paths.len();
    if total_images == 0 {
        println!("❌ Erreur : Aucune image trouvée.");
        return Ok(());
    }
    println!("📸 {} images détectées. Début du traitement...", total_images);

    // Variable pour suivre l'identifiant de la trajectoire en cours
    let mut last_marker_id: Option<i32> = None;

    for (frame_idx, img_path) in image_paths.iter().enumerate() {
        let json_path = img_path.with_extension("json");
        if !json_path.exists() { continue; }
        
        let mut file = File::open(&json_path)?;
        let mut json_str = String::new();
        file.read_to_string(&mut json_str)?;
        let gt_data: GroundTruthData = serde_json::from_str(&json_str)?;

        // --- B. CHARGEMENT DE L'IMAGE BRUTE ---
        let mut img = match image::open(img_path) {
            Ok(i) => i,
            Err(_) => continue,
        };
        let (img_w, img_h) = img.dimensions();

        // --- C. DETERMINATION DE LA ROI ---
        let mut x_coords = Vec::new();
        let mut y_coords = Vec::new();
        for corner in &gt_data.aruco_corners {
            if corner.len() == 2 {
                x_coords.push(corner[0]);
                y_coords.push(corner[1]);
            }
        }
        if x_coords.is_empty() { continue; }

        let xmin = x_coords.iter().cloned().fold(f32::INFINITY, f32::min) as u32;
        let xmax = x_coords.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as u32;
        let ymin = y_coords.iter().cloned().fold(f32::INFINITY, f32::min) as u32;
        let ymax = y_coords.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as u32;

        let center_x = (xmin + xmax) / 2;
        let center_y = (ymin + ymax) / 2;
        let box_w = xmax.saturating_sub(xmin);
        let box_h = ymax.saturating_sub(ymin);
        let margin = (box_w.max(box_h) as f32 * 1.2) as u32;

        let crop_xmin = center_x.saturating_sub(margin).max(0);
        let crop_xmax = (center_x + margin).min(img_w);
        let crop_ymin = center_y.saturating_sub(margin).max(0);
        let crop_ymax = (center_y + margin).min(img_h);

        let crop_w = crop_xmax.saturating_sub(crop_xmin);
        let crop_h = crop_ymax.saturating_sub(crop_ymin);
        if crop_w == 0 || crop_h == 0 { continue; }

        let roi = img.crop(crop_xmin, crop_ymin, crop_w, crop_h);
        let roi_resized = roi.resize_exact(128, 128, image::imageops::FilterType::Triangle);

        // --- D. PREPROCESSING ---
        let mut input_tensor = Array4::<f32>::zeros((1, 3, 128, 128));
        for (x, y, pixel) in roi_resized.pixels() {
            if x < 128 && y < 128 {
                input_tensor[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0; 
                input_tensor[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0; 
                input_tensor[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0; 
            }
        }

        // --- E. INFÉRENCE ---
        let start_inference = Instant::now();
        let shape = vec![1, 3, 128, 128];
        let flat_data = input_tensor.into_raw_vec();
        let input_value = Value::from_array((shape, flat_data))?;
        
        let outputs = session.run(ort::inputs!["input_roi" => input_value])?;
        let output_tensor = outputs["output_pose"].try_extract_tensor::<f32>()?;
        let (_shape, predictions_slice) = output_tensor;
        let inference_duration = start_inference.elapsed();

        let raw_cnn_z  = predictions_slice[2]; 
        let pred_roll  = predictions_slice[3] * 180.0;
        let pred_pitch = predictions_slice[4] * 180.0;

        // ✨ GESTION DU SAUT DE TRAJECTOIRE (RESET KALMAN)
        if let Some(last_id) = last_marker_id {
            if last_id != gt_data.marker_id {
                // Nouveau marqueur détecté -> On réinitialise l'état du filtre sur la mesure brute actuelle
                kalman_z.reset(raw_cnn_z);
            }
        } else {
            // Première frame du traitement complet
            kalman_z.reset(raw_cnn_z);
        }
        last_marker_id = Some(gt_data.marker_id);

        // --- F. FILTRE DE KALMAN ---
        kalman_z.predict(dt);
        kalman_z.update(raw_cnn_z);

        // --- G. AFFICHAGE ÉCHANTILLONNÉ ---
        if frame_idx % 50 == 0 || frame_idx == total_images - 1 {
            let file_name = img_path.file_name().unwrap_or_default().to_string_lossy();
            println!(
                "🎯 Frame [{:04}/{:04}] | ID Trajectoire: {} | T: {:.2?}",
                frame_idx + 1, total_images, gt_data.marker_id, inference_duration
            );
            println!(
                "       📊 [ANGLES] -> Roll Réel: {:.1}° (CNN: {:.1}°) | Pitch Réel: {:.1}° (CNN: {:.1}°)",
                gt_data.roll_deg, pred_roll, gt_data.pitch_deg, pred_pitch
            );
            println!(
                "       🛰️  [ALTITUDE Z] -> Réelle: {:.2}m | Brute CNN: {:.3}m | ✨ Kalman: {:.3}m",
                gt_data.distance_m, raw_cnn_z, kalman_z.x
            );
            println!("--------------------------------------------------------------------------------");
        }
    }

    println!("🏁 Traitement complet terminé !");
    Ok(())
}
