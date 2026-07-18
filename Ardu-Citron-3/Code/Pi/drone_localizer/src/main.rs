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
    pub pitch: f64,
    q_dist: f64,
    r_dist: f64,
    q_pitch: f64,
    r_pitch: f64,
}

impl DroneKalmanFilter {
    pub fn new() -> Self {
        Self { 
            distance: 0.0, 
            pitch: 0.0, 
            q_dist: 0.005,  
            r_dist: 0.08,
            q_pitch: 0.01,  
            r_pitch: 0.12,
        }
    }
    pub fn fuse_and_filter(&mut self, raw_d: f64, raw_p: f64) {
        let k_dist = self.q_dist / (self.q_dist + self.r_dist);
        if self.distance == 0.0 { self.distance = raw_d; } 
        else { self.distance = self.distance + k_dist * (raw_d - self.distance); }

        let k_pitch = self.q_pitch / (self.q_pitch + self.r_pitch);
        if self.pitch == 0.0 { self.pitch = raw_p; } 
        else { self.pitch = self.pitch + k_pitch * (raw_p - self.pitch); }
    }
}

// Extraction mathématique exacte du Pitch via la matrice de Rotation (Rodrigues)
fn get_pitch_from_rvec(rvec: &Mat) -> Result<f64, opencv::Error> {
    let mut rot_mat = Mat::default();
    calib3d::rodrigues(rvec, &mut rot_mat, &mut core::no_array())?;

    let r20 = *rot_mat.at_2d::<f64>(2, 0)?;
    let r21 = *rot_mat.at_2d::<f64>(2, 1)?;
    let r22 = *rot_mat.at_2d::<f64>(2, 2)?;

    // Calcul du pitch signé selon la convention standard OpenCV
    let pitch_rad = (-r20).atan2((r21 * r21 + r22 * r22).sqrt());
    Ok(pitch_rad.to_degrees())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 { std::process::exit(1); }
    
    let dataset_dir = &args[1];
    let max_images = if args.len() > 2 { args[2].parse::<usize>().unwrap_or(50) } else { 50 };

    let folder_pattern = format!("{}/*.png", dataset_dir);
    let mut entries: Vec<PathBuf> = glob::glob(&folder_pattern)?.filter_map(Result::ok).collect();
    entries.sort();

    let camera_matrix = Mat::from_slice_2d(&[[492.72, 0.0, 320.0], [0.0, 492.72, 240.0], [0.0, 0.0, 1.0]])?;
    let dist_coeffs = Mat::from_slice(&[-0.07, 0.0, 0.0, 0.0, 0.0])?;
    
    let h = 0.20 / 2.0;
    let mut obj_points = Vector::<core::Point3d>::new();
    obj_points.push(core::Point3d::new(-h, h, 0.0));
    obj_points.push(core::Point3d::new(h, h, 0.0));
    obj_points.push(core::Point3d::new(h, -h, 0.0));
    obj_points.push(core::Point3d::new(-h, -h, 0.0));

    let dictionary = aruco::get_predefined_dictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50)?;
    
    // 🚀 POINT 4 : Activation du raffinement sub-pixel pour écraser le bruit de quantification
    let mut detector_params = aruco::DetectorParameters::default()?;
    detector_params.corner_refinement_method = aruco::CornerRefineMethod::CORNER_REFINE_SUBPIX;
    let detector_params_ptr = Ptr::new(detector_params);

    let mut kalman = DroneKalmanFilter::new();
    let mut last_center: Option<Point> = None;
    let roi_margin = 110; 

    let mut count = 0;
    for image_path in entries {
        if count >= max_images { break; }

        let file_name = image_path.file_name().unwrap().to_string_lossy().to_string();
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

            // IPPE_SQUARE génère 2 solutions pour contrer l'ambiguïté planaire
            calib3d::solve_pnp_generic(&obj_points, &adjusted_corners, &camera_matrix, &dist_coeffs, &mut rvecs, &mut tvecs, false, calib3d::SolvePnPMethod::SOLVEPNP_IPPE_SQUARE, &Mat::default(), &Mat::default(), &mut core::no_array())?;
            
            let best_tvec = tvecs.get(0)?;
            let best_rvec = rvecs.get(0)?;

            let tx = *best_tvec.at_2d::<f64>(0, 0)?;
            let ty = *best_tvec.at_2d::<f64>(1, 0)?;
            let tz = *best_tvec.at_2d::<f64>(2, 0)?;
            
            let norm_3d = (tx*tx + ty*ty + tz*tz).sqrt();

            // 🚀 POINT 2 : Extraction exacte du Pitch (signé) via Rodrigues
            let raw_pitch = get_pitch_from_rvec(&best_rvec)?;

            kalman.fuse_and_filter(norm_3d, raw_pitch);
            let duration = start_time.elapsed();

            // On envoie à Python : norm_3d, tz (Z pur), et le pitch filtré exact
            println!("DATA|{}|{:.3}|{:.3}|{:.2}|{:.3}", file_name, kalman.distance, tz, kalman.pitch, duration.as_secs_f64() * 1000.0);
            count += 1;
        } else {
            last_center = None;
            println!("NODATA|{}", file_name);
            count += 1;
        }
    }
    Ok(())
}
