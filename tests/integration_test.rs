use python_marshal::magic::PyVersion;
use rayon::prelude::*;
use std::{
    io::{BufReader, Write},
    path::{Path, PathBuf},
};

use pyc_editor::{dump_pyc, load_pyc};

use crate::common::DATA_PATH;

mod common;

#[test]
fn test_recompile_standard_lib() {
    common::setup();
    env_logger::init();

    common::PYTHON_VERSIONS.iter().for_each(|version| {
        println!("Testing with Python version: {}", version);
        let pyc_files = common::find_pyc_files(version);

        pyc_files.iter().for_each(|pyc_file| {
            if pyc_file.ends_with("build_scripts.cpython-310.pyc") {
                println!("Testing pyc file: {:?}", pyc_file);
                let file = std::fs::File::open(&pyc_file).expect("Failed to open pyc file");
                let reader = BufReader::new(file);

                let original_pyc =
                    python_marshal::load_pyc(reader).expect("Failed to load pyc file");
                let original_pyc = python_marshal::resolver::resolve_all_refs(
                    &original_pyc.object,
                    &original_pyc.references,
                )
                .0;

                let file = std::fs::File::open(&pyc_file).expect("Failed to open pyc file");
                let reader = BufReader::new(file);

                let parsed_pyc = load_pyc(reader).unwrap();
                dbg!(&parsed_pyc);
                let pyc: python_marshal::PycFile = parsed_pyc.clone().into();

                std::assert_eq!(
                    original_pyc,
                    pyc.object,
                    "{:?} has not been recompiled succesfully",
                    &pyc_file
                );
            }
        });
    });
}

#[test]
#[ignore = "This test will write the files to disk so we can run the Python tests on them. That way we're sure the files are correct."]
fn test_write_standard_lib() {
    common::setup();
    env_logger::init();

    common::PYTHON_VERSIONS.par_iter().for_each(|version| {
        println!("Testing with Python version: {}", version);
        let pyc_files = common::find_pyc_files(version);

        pyc_files.par_iter().for_each(|pyc_file| {
            println!("Testing pyc file: {:?}", pyc_file);
            let file = std::fs::File::open(&pyc_file).expect("Failed to open pyc file");
            let mut reader = BufReader::new(file);

            let pyc = load_pyc(&mut reader).expect("Failed to read pyc file");

            let output_dir = get_custom_path(&pyc_file.parent().unwrap(), version, "rewritten")
                .parent()
                .unwrap()
                .to_path_buf();

            std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

            let output_path = Path::new(&output_dir).join(pyc_file.file_name().unwrap());

            let mut output_file =
                std::fs::File::create(&output_path).expect("Failed to create output file");

            output_file
                .write(&dump_pyc(pyc).expect("Failed to dump pyc file"))
                .expect(&format!("Failed to write to {:?}", output_path));
        });
    });
}

fn get_custom_path(original_path: &Path, version: &PyVersion, prefix: &'static str) -> PathBuf {
    let relative_path = original_path
        .strip_prefix(Path::new(DATA_PATH).join(format!("cpython-{}/Lib", version)))
        .unwrap();
    Path::new(DATA_PATH)
        .join(format!("{prefix}-{version}/Lib"))
        .join(relative_path)
}
