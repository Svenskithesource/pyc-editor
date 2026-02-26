use core::panic;
use pyc_editor::{
    cfg::{BlockIndexInfo, BranchEdge, ControlFlowGraph},
    dump_pyc, load_pyc,
    prelude::*,
    traits::GenericInstruction,
    utils::ExceptionTableEntry,
    v310, v311, v312, v313,
};

use python_marshal::CodeFlags;
use python_marshal::magic::PyVersion;
use rayon::prelude::*;
use std::{
    io::{BufReader, Write},
    path::{Path, PathBuf},
};

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
    LOGGER_INIT.call_once(|| {
        common::setup();
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

/// We compare the instructions directly instead of the bytes because we need to handle some special cases.
/// For example Python has a bug where it can emit this code:
/// ```python
/// 400    EXTENDED_ARG 1
/// 402    JUMP_BACKWARDS 0
/// ```
/// While this is semantically the same (and what this library outputs):
/// ```python
/// 400    JUMP_BACKWARDS 255
/// ```
///
/// This function handles those discrepancies and treats them as if they were the same code.
/// Returns true if they're equal, false if they're not
fn compare_instructions<T: SimpleInstructionAccess<I>, I>(original_list: T, new_list: T) -> bool {
    let mut og_iter = original_list.as_ref().iter().enumerate();
    let mut new_iter = new_list.as_ref().iter().enumerate();

    // Used to keep track of where the bugs have occured so we can correctly offset other jumps.
    let mut bug_indexes: Vec<usize> = vec![];
    // We can only check for mismatches after we made the full bug index list. So we're saving a list for comparing later.
    let mut possible_mismatches: Vec<(usize, usize)> = vec![]; // (index of original list, index of new list)

    while let Some((og_index, og_instruction)) = og_iter.next() {
        if let Some((new_index, new_instruction)) = new_iter.next() {
            // See the pattern in the doc string. We're trying to pattern match this
            if new_instruction != og_instruction
                && new_instruction.is_jump_backwards()
                && og_instruction.is_extended_arg()
            {
                let mut curr_instruction = og_instruction;

                while curr_instruction.is_extended_arg() {
                    let prev_instruction = curr_instruction;

                    curr_instruction = match og_iter.next() {
                        None => return false,
                        Some((_, new_inst)) => new_inst,
                    };

                    if curr_instruction.is_extended_arg()
                        && prev_instruction.get_raw_value().to_u8() != u8::MAX
                    {
                        // Has to be max value for the bug to happen
                        return false;
                    }
                }

                if !(curr_instruction.is_jump_backwards()
                    && curr_instruction.get_raw_value().to_u8() == 0)
                {
                    // Bug did not occur so there could be an actual mismatch or a difference in indexes due to previous bugs
                    possible_mismatches.push((og_index, new_index));
                } else {
                    // Bug occured
                    bug_indexes.push(new_index);
                }
            } else if new_instruction != og_instruction
                && new_instruction.get_opcode() == og_instruction.get_opcode()
                && new_instruction.is_jump()
            {
                // If both are the same jump opcode, their jump target indexes could differ due to the bug being triggered
                possible_mismatches.push((og_index, new_index));
            } else if new_instruction != og_instruction {
                return false;
            }
        } else {
            // Length of the instructions doesn't match
            return false;
        }
    }

    // Check if they're actual mismatches or caused by bugs
    for (og_index, new_index) in possible_mismatches {
        let bug_count = if new_list.as_ref().get(new_index).unwrap().is_absolute_jump() {
            bug_indexes
                .iter()
                .filter(|bug_index| **bug_index < new_index)
                .count()
        } else if new_list.as_ref().get(new_index).unwrap().is_jump_forwards() {
            bug_indexes
                .iter()
                .filter(|bug_index| {
                    new_index < **bug_index
                        && **bug_index
                            < new_index + new_list.get_full_arg(new_index).unwrap() as usize + 1
                })
                .count()
        } else if new_list
            .as_ref()
            .get(new_index)
            .unwrap()
            .is_jump_backwards()
        {
            bug_indexes
                .iter()
                .filter(|bug_index| {
                    new_index - new_list.get_full_arg(new_index).unwrap() as usize + 1 < **bug_index
                        && **bug_index < new_index
                })
                .count()
        } else {
            unreachable!("It should always be a jump. We have covered all jump types.")
        };

        if new_list.get_full_arg(new_index).unwrap() + bug_count as u32
            != original_list.get_full_arg(og_index).unwrap()
        {
            return false;
        }
    }

    true
}

#[test]
fn test_recompile_resolved_standard_lib() {
    LOGGER_INIT.call_once(|| {
        common::setup();
        env_logger::init();
    });

    common::PYTHON_VERSIONS.par_iter().for_each(|version| {
        println!("Testing with Python version: {}", version);
        let pyc_files = common::find_pyc_files(version);

        pyc_files.par_iter().for_each(|pyc_file| {
            println!("Testing pyc file: {:?}", pyc_file);
            let file = std::fs::File::open(pyc_file).expect("Failed to open pyc file");
            let reader = BufReader::new(file);

            let mut parsed_pyc = load_pyc(reader).unwrap();

            fn rewrite_code_object(
                code: pyc_editor::CodeObject,
            ) -> Result<pyc_editor::CodeObject, (pyc_editor::CodeObject, pyc_editor::CodeObject)>
            {
                macro_rules! rewrite_version {
                    ($variant:ident, $module:ident, $code:expr) => {{
                        let mut code = $code.clone();
                        let mut new_code = $code.clone();
                        new_code.code = new_code.code.to_resolved().unwrap().to_instructions();
                        match compare_instructions(
                            $code.code.iter().as_slice(),
                            new_code.code.iter().as_slice(),
                        ) {
                            false => {
                                return Err((
                                    pyc_editor::CodeObject::$variant(code),
                                    pyc_editor::CodeObject::$variant(new_code),
                                ));
                            }
                            true => {}
                        }

                        for constant in &mut code.consts {
                            if let &mut $module::code_objects::Constant::CodeObject(
                                ref mut const_code,
                            ) = constant
                            {
                                rewrite_code_object(pyc_editor::CodeObject::$variant(
                                    const_code.clone(),
                                ))?;
                            }
                        }

                        Ok(pyc_editor::CodeObject::$variant(new_code))
                    }};
                }

                handle_code_object_versions!(code, rewrite_version)
            }

            macro_rules! rewrite_pyc {
                ($variant:ident, $module:ident, $pyc:expr) => {{
                    match rewrite_code_object(pyc_editor::CodeObject::$variant(
                        $pyc.code_object.clone(),
                    )) {
                        Ok(new_code) => new_code,
                        Err((
                            pyc_editor::CodeObject::$variant(code),
                            pyc_editor::CodeObject::$variant(new_code),
                        )) => {
                            println!("{:#?}", code);
                            println!("{:#?}", new_code);
                            panic!();
                        }
                        _ => unreachable!(),
                    };
                }};
            }

            handle_pyc_versions!(parsed_pyc, rewrite_pyc);
        });
    });
}

#[test]
fn test_line_number_standard_lib() {
    LOGGER_INIT.call_once(|| {
        common::setup();
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
                            if let &$module::code_objects::Constant::CodeObject(ref const_code) =
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
fn test_stacksize_standard_lib() {
    LOGGER_INIT.call_once(|| {
        common::setup();
        env_logger::init();
    });

    // Cpython has a bug where it overcalculates the stacksize of these files, so we skip it.
    static EXCEPTIONS: &[&str] = &[
        "tests/data/cpython-3.11.1/Lib/test/__pycache__/test_except_star.cpython-311.pyc",
        "tests/data/cpython-3.11.1/Lib/test/__pycache__/test_sys_settrace.cpython-311.pyc",
        "tests/data/cpython-3.11.1/Lib/__pycache__/opcode.cpython-311.pyc", // Only fails on github actions for some reason
        "tests/data/cpython-3.12.1/Lib/test/__pycache__/test_except_star.cpython-312.pyc",
        "tests/data/cpython-3.12.1/Lib/test/__pycache__/test_sys_settrace.cpython-312.pyc",
        "tests/data/cpython-3.13.1/Lib/test/__pycache__/test_except_star.cpython-313.pyc",
        "tests/data/cpython-3.13.1/Lib/test/__pycache__/test_sys_settrace.cpython-313.pyc",
    ];

    common::PYTHON_VERSIONS.par_iter().for_each(|version| {
        println!("Testing with Python version: {}", version);
        let pyc_files = common::find_pyc_files(version);

        pyc_files.par_iter().for_each(|pyc_file| {
            if EXCEPTIONS.iter().all(|exc| !pyc_file.ends_with(exc)) {
                println!("Testing pyc file: {:?}", pyc_file);
                let file = std::fs::File::open(pyc_file).expect("Failed to open pyc file");
                let reader = BufReader::new(file);

                let parsed_pyc = load_pyc(reader).unwrap();

                fn recursive_code_object(code: &pyc_editor::CodeObject) {
                    macro_rules! get_exception_table {
                        (V310, $code:expr) => {
                            None
                        };

                        ($variant:ident, $code:expr) => {
                            Some($code.exception_table().expect("Exception table is valid"))
                        };
                    }

                    macro_rules! get_generator_stacksize {
                        (V313, $code:expr) => {
                            0
                        };

                        ($variant:ident, $code:expr) => {
                            1
                        };
                    }

                    macro_rules! allow_zero {
                        (V313, $code:expr) => {
                            false
                        };

                        ($variant:ident, $code:expr) => {
                            true
                        };
                    }

                    macro_rules! recursive_version {
                        ($variant:ident, $module:ident, $code:expr) => {{
                            let is_generator = $code.flags.intersects(
                                CodeFlags::GENERATOR
                                    | CodeFlags::COROUTINE
                                    | CodeFlags::ASYNC_GENERATOR
                                    | CodeFlags::ITERABLE_COROUTINE,
                            );

                            let exception_table: Option<Vec<ExceptionTableEntry>> =
                                get_exception_table!($variant, $code);

                            assert_eq!(
                                $code
                                    .code
                                    .max_stack_size(
                                        if is_generator {
                                            get_generator_stacksize!($variant, $code)
                                        } else {
                                            0
                                        },
                                        exception_table,
                                        allow_zero!($variant, $code)
                                    )
                                    .expect("Must be valid"),
                                $code.stacksize
                            );

                            for constant in &$code.consts {
                                if let &$module::code_objects::Constant::CodeObject(
                                    ref const_code,
                                ) = constant
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
            }
        });
    });
}

#[test]
fn test_create_cfg_standard_lib() {
    use pyc_editor::cfg::create_cfg;

    fn get_len_without_cache<I>(instructions: &[I]) -> usize
    where
        I: GenericInstruction,
    {
        instructions.iter().filter(|i| !i.is_cache()).count()
    }

    fn count_cfg_instructions<I, BranchReason>(cfg: ControlFlowGraph<I, BranchReason>) -> usize
    where
        I: GenericInstruction,
        BranchReason: BranchReasonTrait,
    {
        let mut instruction_count = 0;

        for block in cfg.blocks {
            instruction_count += get_len_without_cache(&block.instructions);

            match block.branch_block {
                BlockIndexInfo::Edge(BranchEdge { reason, .. }) => {
                    if reason.is_opcode() {
                        instruction_count += 1;
                    }
                }
                _ => {}
            }
        }

        instruction_count
    }

    LOGGER_INIT.call_once(|| {
        common::setup();
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

            fn create_cfg_from_code(code: pyc_editor::CodeObject) {
                macro_rules! create_cfg {
                    (V310, $code_clone:ident) => {
                        create_cfg::<_, _, pyc_editor::v310::opcodes::Opcode>(
                            $code_clone.code.to_vec(),
                            None,
                        )
                    };
                    (V311, $code_clone:ident) => {
                        create_cfg::<_, _, pyc_editor::v311::opcodes::BranchReason>(
                            $code_clone.code.to_vec(),
                            Some($code_clone.exception_table().unwrap()),
                        )
                    };
                    (V312, $code_clone:ident) => {
                        create_cfg::<_, _, pyc_editor::v312::opcodes::BranchReason>(
                            $code_clone.code.to_vec(),
                            Some($code_clone.exception_table().unwrap()),
                        )
                    };
                    (V313, $code_clone:ident) => {
                        create_cfg::<_, _, pyc_editor::v313::opcodes::BranchReason>(
                            $code_clone.code.to_vec(),
                            Some($code_clone.exception_table().unwrap()),
                        )
                    };
                }

                macro_rules! cfg_from_instructions {
                    ($variant:ident, $module:ident, $code:expr) => {{
                        let code_clone = $code.clone();

                        let cfg = create_cfg!($variant, code_clone).unwrap();

                        assert_eq!(
                            count_cfg_instructions(cfg),
                            get_len_without_cache(&code_clone.code)
                        );

                        for constant in code_clone.consts {
                            if let $module::code_objects::Constant::CodeObject(const_code) =
                                constant
                            {
                                create_cfg_from_code(pyc_editor::CodeObject::$variant(
                                    const_code.clone(),
                                ));
                            }
                        }
                    }};
                }

                handle_code_object_versions!(code, cfg_from_instructions)
            }

            macro_rules! create_cfgs {
                ($variant:ident, $module:ident, $pyc:expr) => {{
                    create_cfg_from_code(pyc_editor::CodeObject::$variant(
                        $pyc.code_object.clone(),
                    ));
                }};
            }

            handle_pyc_versions!(parsed_pyc, create_cfgs, immutable);
        });
    });
}

#[test]
#[ignore = "This test will write the files to disk so we can run the Python tests on them. That way we're sure the files are correct."]
fn test_write_standard_lib() {
    LOGGER_INIT.call_once(|| {
        common::setup();
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
