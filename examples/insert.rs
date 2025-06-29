use pyc_editor::{
    dump_pyc, load_pyc,
    v310::{code_objects::Instruction, opcodes::Opcode},
    PycFile,
};

fn main() {
    let data = include_bytes!("./pyc_files/modify.cpython-310.pyc");

    let mut pyc_file = load_pyc(data.as_slice()).expect("Invalid pyc file");

    match pyc_file {
        PycFile::V310(ref mut pyc) => {
            // Call print(a + b) twice
            let code = &mut pyc.code_object.code;
            let (index, &call) = code
                .iter()
                .enumerate()
                .find(|(_, i)| i.get_opcode() == Opcode::CALL_FUNCTION)
                .expect("Call not found");

            code.insert_instructions(
                index,
                &[
                    Instruction::DupTop, // Duplicate argument for the call (a + b)
                    call,                // Call print for the second time
                ],
            );
        }
    }

    dbg!(&pyc_file); // Addition is changed to subtraction

    // You can now write it back to bytes (which can be directly to a file)
    let mut out = Vec::new();
    dump_pyc(&mut out, pyc_file).expect("Invalid pyc file");
}
