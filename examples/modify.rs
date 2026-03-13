use pyc_editor::{
    PycFile, dump_pyc, load_pyc,
    prelude::*,
    v310::{instructions::Instruction, opcodes::Opcode},
};

fn main() {
    let data = include_bytes!("./pyc_files/modify.cpython-310.pyc");

    let mut pyc_file = load_pyc(data.as_slice()).expect("Invalid pyc file");

    if let PycFile::V310(ref mut pyc) = pyc_file {
        // Change print(a + b) to print(a - b)
        let add = pyc
            .code_object
            .code
            .iter_mut()
            .find(|i| i.get_opcode() == Opcode::BINARY_ADD)
            .expect("Add opcode not found");

        *add = Instruction::BinarySubtract(0);
    }

    dbg!(&pyc_file); // Addition is changed to subtraction

    // You can now write it back to bytes
    let _out = dump_pyc(pyc_file).expect("Invalid pyc file");
}
