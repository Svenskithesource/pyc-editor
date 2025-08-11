use pyc_editor::{
    dump_pyc, load_pyc,
    prelude::*,
    v310::{ext_instructions::ExtInstruction, opcodes::Opcode},
    PycFile,
};

fn main() {
    let data = include_bytes!("./pyc_files/modify.cpython-310.pyc");

    let mut pyc_file = load_pyc(data.as_slice()).expect("Invalid pyc file");

    match pyc_file {
        PycFile::V310(ref mut pyc) => {
            // Call print(a + b) twice
            let mut resolved = pyc.code_object.code.to_resolved();
            let (index, &call) = resolved
                .iter()
                .enumerate()
                .find(|(_, i)| i.get_opcode() == Opcode::CALL_FUNCTION)
                .expect("Call not found");

            resolved.insert_instructions(
                index,
                &[
                    ExtInstruction::DupTop(0.into()), // Duplicate argument for the call (a + b)
                    call,                             // Call print for the second time
                ],
            );

            pyc.code_object.code = resolved.to_instructions();
        }
    }

    dbg!(&pyc_file); // Addition is changed to subtraction

    // You can now write it back to bytes
    let _out = dump_pyc(pyc_file).expect("Invalid pyc file");
}
