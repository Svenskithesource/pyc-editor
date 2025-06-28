use rayon::prelude::*;
use std::{
    io::BufReader,
};

use pyc_editor::load_pyc;

mod common;

fn delete_debug_files() {
    let _ = std::fs::remove_file("debug_output.txt");
    let _ = std::fs::remove_file("write_log.txt");
    let _ = std::fs::remove_file("read_log.txt");
}

#[test]
fn test_recompile_standard_lib() {
    common::setup();
    env_logger::init();
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build_global()
    //     .unwrap();

    common::PYTHON_VERSIONS.par_iter().for_each(|version| {
        println!("Testing with Python version: {}", version);
        let pyc_files = common::find_pyc_files(version);

        pyc_files.par_iter().for_each(|pyc_file| {
            delete_debug_files();
            println!("Testing pyc file: {:?}", pyc_file);
            let file = std::fs::File::open(&pyc_file).expect("Failed to open pyc file");
            let reader = BufReader::new(file);

            let original_pyc = python_marshal::load_pyc(reader).expect("Failed to load pyc file");
            let original_pyc = python_marshal::resolver::resolve_all_refs(
                original_pyc.object,
                original_pyc.references,
            )
            .unwrap()
            .0;

            let file = std::fs::File::open(&pyc_file).expect("Failed to open pyc file");
            let reader = BufReader::new(file);

            let parsed_pyc = load_pyc(reader).unwrap();
            let pyc: python_marshal::PycFile = parsed_pyc.into();

            std::assert_eq!(
                original_pyc,
                pyc.object,
                "{:?} has not been recompiled succesfully",
                &pyc_file
            );
        });
    });
}
