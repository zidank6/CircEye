// Prevents console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

fn main() {
    // Initialize Tauri with plugins and commands
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .invoke_handler(tauri::generate_handler![commands::save_visualization])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
