use opencv::{
    calib3d,
    core::{self, Mat, Point2f, Vector, Ptr, Vec3b, Rect, Point},
    imgproc,
    aruco,
    prelude::*,
};
use std::time::Instant;
use std::path::PathBuf;
use std::env;

pub struct DroneKalmanFilter {
    pub distance: f64,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
    
    q_dist: f64, r_dist: f64,
    q_angle: f64, r_angle: f64,
}

impl DroneKalmanFilter {
    pub fn new() -> Self {
        Self { 
            distance: 0.0, roll: 0.0, pitch: 0.0, yaw: 0.0, 
            q_dist: 0.002, r_dist: 0.05,
            q_angle: 0.05, r_angle: 0.20,
        }
    }

    pub fn fuse_all(&mut self, raw_d: f64, raw_r: f64, raw_p: f64, raw_y: f64) {
        let k_dist = self.q_dist / (self.q_dist + self.r_dist);
        if self.distance == 0.0 { self.distance = raw_d; } 
        else { self.distance = self.distance + k_dist * (raw_d - self.distance); }

        let k_a = self.q_angle / (self.q_angle + self.r_angle);
        
        if self.pitch == 0.0 && self.roll == 0.0 {
            self.roll = raw_r; self.pitch = raw_p; self.yaw = raw_y;
        } else {
            let mut diff_roll = raw_r - self.roll;
            if diff_roll > 180.0 { diff_roll -= 360.0; }
            if diff_roll < -180.0 { diff_roll += 360.0; }
            self.roll = self.roll + k_a * diff_roll;

            let mut diff_pitch = raw_p - self.pitch;
            if diff_pitch > 180.0 { diff_pitch -= 360.0; }
            if diff_pitch < -180.0 { diff_pitch += 360.0; }
            self.pitch = self.pitch + k_a * diff_pitch;

            let mut diff_yaw = raw_y - self.yaw;
            if diff_yaw > 180.0 { diff_yaw -= 360.0; }
            if diff_yaw < -180.0 { diff_yaw += 360.0; }
            self.yaw = self.yaw + k_a * diff_yaw;
        }
    }
}

// Extraction adaptée au repère d'une caméra pointant vers le bas (Ventrale)
fn get_drone_angles_from_camera_matrix(r: &Mat) -> Result<(f64, f64, f64), opencv::Error> {
    // Extraction des coefficients de la matrice OpenCV standard
    let r00 = *r.at_2d::<f64>(0, 0)?;
    let r01 = *r.at_2d::<f64>(0, 1)?;
    let r02 = *r.at_2d::<f64>(0, 2)?;
    let r10 = *r.at_2d::<f64>(1, 0)?;
    let r11 = *r.at_2d::<f64>(1, 1)?;
    let r12 = *r.at_2d::<f64>(1, 2)?;
    let r20 = *r.at_2d::<f64>(2, 0)?;
    let r21 = *r.at_2d::<f64>(2, 1)?;
    let r22 = *r.at_2d::<f64>(2, 2)?;

    // Transformation de repère : Passage de la matrice caméra au repère corps du drone
    // On réaligne la rotation selon la convention NED (North-East-Down) de ton simulateur
    let pitch = (-r20).atan2((r21 * r21 + r22 * r22).sqrt()).to_degrees();
    let roll = r21.atan2(r22).to_degrees();
    let yaw = r10.atan2(r00).to_degrees();

    // Application du complément géométrique pour la caméra ventrale inversée
    let corrected_roll = -roll; 
    let corrected_pitch = -pitch;

    Ok((corrected_roll, corrected_pitch, yaw))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 { std::process::exit(1); }
    
    let dataset_dir = &args[1];
    let max_images = if args.len() > 2 { args[2].parse::<usize>().unwrap_or(50) } else { 50 };

    let folder_pattern = format!("{}/*.png", dataset_dir);
    let mut entries: Vec<PathBuf> = glob::glob(&folder_pattern)?.filter_map(Result::ok).collect();
    entries.sort();

    let camera_matrix = Mat::from_slice_2d(&[
        [492.7568, 0.0, 320.0], [0.0, 492.0729, 240.0], [0.0, 0.0, 1.0]
    ])?;
    
    let dist_coeffs = Mat::from_slice(&[-0.07, 0.0, 0.0, 0.0, 0.0])?;
    
    let h = 0.20 / 2.0;
    let mut obj_points = Vector::<core::Point3d>::new();
    obj_points.push(core::Point3d::new(-h, h, 0.0));
    obj_points.push(core::Point3d::new(h, h, 0.0));
    obj_points.push(core::Point3d::new(h, -h, 0.0));
    obj_points.push(core::Point3d::new(-h, -h, 0.0));

    let dictionary = aruco::get_predefined_dictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50)?;
    let mut detector_params = aruco::DetectorParameters::default()?;
    detector_params.set_corner_refinement_method(aruco::CornerRefineMethod::CORNER_REFINE_SUBPIX as i32);
    let detector_params_ptr = Ptr::new(detector_params);

    let mut kalman = DroneKalmanFilter::new();
    let mut last_center: Option<Point> = None;
    let roi_margin = 110; 

    for image_path in entries.into_iter().take(max_images) {
        let file_name = image_path.file_name().unwrap().to_string_lossy().to_string();
        
        // On ne lit plus du tout les angles dans le JSON pour s'aligner ! Autonomie totale.
        let img_decoded = match image::open(&image_path) {
            Ok(img) => img.to_rgb8(),
            Err(_) => continue,
        };
        let (width, height) = img_decoded.dimensions();
        let start_time = Instant::now();
        
        let frame_rgb = Mat::new_rows_cols_with_data::<Vec3b>(
            height as i32, width as i32,
            unsafe { std::slice::from_raw_parts(img_decoded.as_raw().as_ptr() as *const Vec3b, (width * height) as usize) }
        )?;

        let mut gray = Mat::default();
        imgproc::cvt_color(&frame_rgb, &mut gray, imgproc::COLOR_RGB2GRAY, 0)?;

        let mut corners = Vector::<Vector<Point2f>>::new();
        let mut ids = Vector::<i32>::new();
        let mut rejected = Vector::<Vector<Point2f>>::new();
        
        let mut roi_rect = Rect::new(0, 0, width as i32, height as i32);
        let mut using_roi = false;

        if let Some(center) = last_center {
            let x = (center.x - roi_margin).max(0);
            let y = (center.y - roi_margin).max(0);
            let w = (center.x + roi_margin).min(width as i32) - x;
            let h_roi = (center.y + roi_margin).min(height as i32) - y;
            roi_rect = Rect::new(x, y, w, h_roi);
            using_roi = true;
        }

        let gray_roi = Mat::roi(&gray, roi_rect)?;
        aruco::detect_markers(&gray_roi, &dictionary, &mut corners, &mut ids, &detector_params_ptr, &mut rejected)?;

        if ids.is_empty() && using_roi {
            roi_rect = Rect::new(0, 0, width as i32, height as i32);
            aruco::detect_markers(&gray, &dictionary, &mut corners, &mut ids, &detector_params_ptr, &mut rejected)?;
        }

        if !ids.is_empty() {
            let raw_corners = corners.get(0)?;
            let mut adjusted_corners = Vector::<Point2f>::new();
            let mut cx = 0.0; let mut cy = 0.0;

            for i in 0..4 {
                let pt = raw_corners.get(i)?;
                let global_x = pt.x + roi_rect.x as f32;
                let global_y = pt.y + roi_rect.y as f32;
                adjusted_corners.push(Point2f::new(global_x, global_y));
                cx += global_x; cy += global_y;
            }
            
            last_center = Some(Point::new((cx / 4.0) as i32, (cy / 4.0) as i32));

            let mut rvecs = Vector::<Mat>::new();
            let mut tvecs = Vector::<Mat>::new();

            calib3d::solve_pnp_generic(&obj_points, &adjusted_corners, &camera_matrix, &dist_coeffs, &mut rvecs, &mut tvecs, false, calib3d::SolvePnPMethod::SOLVEPNP_IPPE_SQUARE, &Mat::default(), &Mat::default(), &mut core::no_array())?;
            
            let mut r_cam = Mat::default();
            calib3d::rodrigues(&rvecs.get(0)?, &mut r_cam, &mut core::no_array())?;
            
            // Calcul direct et robuste des angles du drone depuis la matrice de rotation OpenCV
            let (raw_r, raw_p, raw_y) = get_drone_angles_from_camera_matrix(&r_cam)?;

            let best_tvec = tvecs.get(0)?;
            let tz = *best_tvec.at_2d::<f64>(2, 0)?;

            kalman.fuse_all(tz.abs(), raw_r, raw_p, raw_y);
            let duration = start_time.elapsed();

            println!("DATA|{}|{:.3}|{:.3}|{:.2}|{:.2}|{:.2}|{:.3}", 
                file_name, kalman.distance, kalman.distance, kalman.roll, kalman.pitch, kalman.yaw, duration.as_secs_f64() * 1000.0);
        } else {
            last_center = None;
            println!("NODATA|{}", file_name);
        }
    }
    Ok(())
}
