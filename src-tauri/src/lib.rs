use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct SaveResult {
    pub success: bool,
    pub path: String,
}

// Saves visualization data to disk via native file system
#[tauri::command]
pub fn save_visualization(path: String, data: Vec<u8>) -> Result<SaveResult, String> {
    let path_buf = PathBuf::from(&path);

    fs::write(&path_buf, &data)
        .map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(SaveResult {
        success: true,
        path: path_buf.to_string_lossy().to_string(),
    })
}
