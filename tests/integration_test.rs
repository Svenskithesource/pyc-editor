use python_marshal::magic::PyVersion;
use rayon::prelude::*;
use std::{
    io::{BufReader, Write},
    path::{Path, PathBuf},
};

use pyc_editor::{dump_pyc, load_pyc, v310, v311, v312, v313};

use crate::common::DATA_PATH;

mod common;

/// Macro to generate version-specific code object handling
macro_rules! handle_code_object_versions {
    ($code:expr, $handler:ident) => {
        match $code {
            pyc_editor::CodeObject::V310(code) => {
                $handler!(V310, v310, code)
            }
            pyc_editor::CodeObject::V311(code) => {
                $handler!(V311, v311, code)
            }
            pyc_editor::CodeObject::V312(code) => {
                $handler!(V312, v312, code)
            }
            pyc_editor::CodeObject::V313(code) => {
                $handler!(V313, v313, code)
            }
        }
    };
}

/// Macro to generate version-specific PYC file handling
macro_rules! handle_pyc_versions {
    ($pyc:expr, $handler:ident) => {
        match $pyc {
            pyc_editor::PycFile::V310(ref mut pyc) => {
                $handler!(V310, v310, pyc)
            }
            pyc_editor::PycFile::V311(ref mut pyc) => {
                $handler!(V311, v311, pyc)
            }
            pyc_editor::PycFile::V312(ref mut pyc) => {
                $handler!(V312, v312, pyc)
            }
            pyc_editor::PycFile::V313(ref mut pyc) => {
                $handler!(V313, v313, pyc)
            }
        }
    };
    // Variant for immutable references
    ($pyc:expr, $handler:ident, immutable) => {
        match $pyc {
            pyc_editor::PycFile::V310(ref pyc) => {
                $handler!(V310, v310, pyc)
            }
            pyc_editor::PycFile::V311(ref pyc) => {
                $handler!(V311, v311, pyc)
            }
            pyc_editor::PycFile::V312(ref pyc) => {
                $handler!(V312, v312, pyc)
            }
            pyc_editor::PycFile::V313(ref pyc) => {
                $handler!(V313, v313, pyc)
            }
        }
    };
}

static LOGGER_INIT: std::sync::Once = std::sync::Once::new();

#[test]
fn test_recompile_standard_lib() {
    common::setup();
    LOGGER_INIT.call_once(|| {
        env_logger::init();
    });

    common::PYTHON_VERSIONS.par_iter().for_each(|version| {
        println!("Testing with Python version: {}", version);
        let pyc_files = common::find_pyc_files(version);

        pyc_files.par_iter().for_each(|pyc_file| {
            println!("Testing pyc file: {:?}", pyc_file);
            let file = std::fs::File::open(pyc_file).expect("Failed to open pyc file");
            let reader = BufReader::new(file);

            let original_pyc = python_marshal::load_pyc(reader).expect("Failed to load pyc file");
            let original_pyc = python_marshal::resolver::resolve_all_refs(
                &original_pyc.object,
                &original_pyc.references,
            )
            .0;

            let file = std::fs::File::open(pyc_file).expect("Failed to open pyc file");
            let reader = BufReader::new(file);

            let parsed_pyc = load_pyc(reader).unwrap();

            let pyc: python_marshal::PycFile = parsed_pyc.clone().into();

            std::assert_eq!(
                original_pyc,
                pyc.object,
                "{:?} has not been recompiled succesfully",
                &pyc_file
            );
        });
    });
}

#[test]
fn test_recompile_resolved_standard_lib() {
    common::setup();
    LOGGER_INIT.call_once(|| {
        env_logger::init();
    });

    common::PYTHON_VERSIONS.iter().for_each(|version| {
        println!("Testing with Python version: {}", version);
        let pyc_files = common::find_pyc_files(version);

        pyc_files.iter().for_each(|pyc_file| {
            println!("Testing pyc file: {:?}", pyc_file);
            let file = std::fs::File::open(pyc_file).expect("Failed to open pyc file");
            let reader = BufReader::new(file);

            let original_pyc = python_marshal::load_pyc(reader).expect("Failed to load pyc file");
            let original_pyc = python_marshal::resolver::resolve_all_refs(
                &original_pyc.object,
                &original_pyc.references,
            )
            .0;

            let file = std::fs::File::open(pyc_file).expect("Failed to open pyc file");
            let reader = BufReader::new(file);

            let mut parsed_pyc = load_pyc(reader).unwrap();

            fn rewrite_code_object(code: &pyc_editor::CodeObject) -> pyc_editor::CodeObject {
                macro_rules! rewrite_version {
                    ($variant:ident, $module:ident, $code:expr) => {{
                        let mut code = $code.clone();
                        code.code = code.code.to_resolved().to_instructions();

                        for constant in &mut code.consts {
                            if let $module::code_objects::Constant::CodeObject(ref mut const_code) =
                                constant
                            {
                                let new_const_code = rewrite_code_object(
                                    &pyc_editor::CodeObject::$variant(const_code.clone()),
                                );

                                if let pyc_editor::CodeObject::$variant(inner_const_code) =
                                    new_const_code
                                {
                                    *const_code = inner_const_code;
                                }
                            }
                        }

                        pyc_editor::CodeObject::$variant(code)
                    }};
                }

                handle_code_object_versions!(code, rewrite_version)
            }

            macro_rules! rewrite_pyc {
                ($variant:ident, $module:ident, $pyc:expr) => {{
                    let new_code = rewrite_code_object(&pyc_editor::CodeObject::$variant(
                        $pyc.code_object.clone(),
                    ));

                    if let pyc_editor::CodeObject::$variant(inner_code) = new_code {
                        $pyc.code_object = inner_code;
                    }
                }};
            }

            handle_pyc_versions!(parsed_pyc, rewrite_pyc);

            let pyc: python_marshal::PycFile = parsed_pyc.clone().into();

            std::assert_eq!(
                original_pyc,
                pyc.object,
                "{:?} has not been recompiled succesfully",
                &pyc_file
            );
        });
    });
}

#[test]
fn test_line_number_standard_lib() {
    common::setup();
    LOGGER_INIT.call_once(|| {
        env_logger::init();
    });

    common::PYTHON_VERSIONS.par_iter().for_each(|version| {
        println!("Testing with Python version: {}", version);
        let pyc_files = common::find_pyc_files(version);

        pyc_files.par_iter().for_each(|pyc_file| {
            println!("Testing pyc file: {:?}", pyc_file);

            let file = std::fs::File::open(pyc_file).expect("Failed to open pyc file");
            let reader = BufReader::new(file);

            let parsed_pyc = load_pyc(reader).unwrap();

            fn recursive_code_object(code: &pyc_editor::CodeObject) {
                macro_rules! recursive_version {
                    ($variant:ident, $module:ident, $code:expr) => {{
                        let co_lines = $code.co_lines().unwrap();
                        for index in 0..$code.code.len() {
                            $module::instructions::get_line_number(&co_lines, index as u32);
                        }

                        for constant in &$code.consts {
                            if let $module::code_objects::Constant::CodeObject(ref const_code) =
                                constant
                            {
                                recursive_code_object(&pyc_editor::CodeObject::$variant(
                                    const_code.clone(),
                                ));
                            }
                        }
                    }};
                }

                handle_code_object_versions!(code, recursive_version);
            }

            macro_rules! call_recursive {
                ($variant:ident, $module:ident, $pyc:expr) => {{
                    recursive_code_object(&pyc_editor::CodeObject::$variant(
                        $pyc.code_object.clone(),
                    ));
                }};
            }

            handle_pyc_versions!(parsed_pyc, call_recursive, immutable);
        });
    });
}

#[test]
#[ignore = "This test will write the files to disk so we can run the Python tests on them. That way we're sure the files are correct."]
fn test_write_standard_lib() {
    common::setup();
    LOGGER_INIT.call_once(|| {
        env_logger::init();
    });

    common::PYTHON_VERSIONS.par_iter().for_each(|version| {
        println!("Testing with Python version: {}", version);
        let pyc_files = common::find_pyc_files(version);

        pyc_files.par_iter().for_each(|pyc_file| {
            println!("Testing pyc file: {:?}", pyc_file);
            let file = std::fs::File::open(pyc_file).expect("Failed to open pyc file");
            let mut reader = BufReader::new(file);

            let pyc = load_pyc(&mut reader).expect("Failed to read pyc file");

            let output_dir = get_custom_path(pyc_file.parent().unwrap(), version, "rewritten")
                .parent()
                .unwrap()
                .to_path_buf();

            std::fs::create_dir_all(&output_dir).expect("Failed to create output directory");

            let output_path = Path::new(&output_dir).join(pyc_file.file_name().unwrap());

            let mut output_file =
                std::fs::File::create(&output_path).expect("Failed to create output file");

            output_file
                .write_all(&dump_pyc(pyc).expect("Failed to dump pyc file"))
                .unwrap_or_else(|_| panic!("Failed to write to {:?}", output_path));
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
