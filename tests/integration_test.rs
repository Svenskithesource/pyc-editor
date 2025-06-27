use common::DATA_PATH;
use rayon::prelude::*;
use std::{
    io::BufReader,
    path::{Path, PathBuf},
};

use pretty_assertions::assert_eq;
use pyc_editor::load_pyc;
use python_marshal::magic::PyVersion;

mod common;

fn delete_debug_files() {
    let _ = std::fs::remove_file("debug_output.txt");
    let _ = std::fs::remove_file("write_log.txt");
    let _ = std::fs::remove_file("read_log.txt");
}

fn diff_bytearrays(a: &[u8], b: &[u8]) -> Vec<(usize, u8, u8)> {
    let mut diff = Vec::new();
    for (i, (&byte_a, &byte_b)) in a.iter().zip(b.iter()).enumerate() {
        if byte_a != byte_b {
            diff.push((i, byte_a, byte_b));
        }
    }
    diff
}

#[test]
fn test_recompile_standard_lib() {
    common::setup();
    env_logger::init();
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();

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

            match (original_pyc.clone(), pyc.object.clone()) {
                (
                    python_marshal::Object::Code(python_marshal::Code::V310(old_code)),
                    python_marshal::Object::Code(python_marshal::Code::V310(new_code)),
                ) => {
                    assert_eq!(old_code.argcount, new_code.argcount);
                    assert_eq!(old_code.posonlyargcount, new_code.posonlyargcount);
                    assert_eq!(old_code.kwonlyargcount, new_code.kwonlyargcount);
                    assert_eq!(old_code.nlocals, new_code.nlocals);
                    assert_eq!(old_code.stacksize, new_code.stacksize);
                    assert_eq!(old_code.flags, new_code.flags);
                    assert_eq!(old_code.code, new_code.code);
                    assert_eq!(old_code.names, new_code.names);
                    assert_eq!(old_code.varnames, new_code.varnames);
                    assert_eq!(old_code.freevars, new_code.freevars);
                    assert_eq!(old_code.cellvars, new_code.cellvars);
                    assert_eq!(old_code.filename, new_code.filename);
                    assert_eq!(old_code.name, new_code.name);
                    assert_eq!(old_code.firstlineno, new_code.firstlineno);
                    assert_eq!(old_code.lnotab, new_code.lnotab);
                }
                _ => assert!(false, "Pyc file does not contain a code object."),
            }

            std::assert_eq!(
                original_pyc, pyc.object,
                "{:?} has not been recompiled succesfully",
                &pyc_file
            );
        });
    });
}
