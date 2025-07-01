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
            // Change print(a + b) to print(a - b)
            let add = pyc
                .code_object
                .code
                .iter_mut()
                .find(|i| i.get_opcode() == Opcode::BINARY_ADD)
                .expect("Add opcode not found");

            *add = Instruction::BinarySubtract;
        }
    }

    dbg!(&pyc_file); // Addition is changed to subtraction

    // You can now write it back to bytes
    let _out = dump_pyc(pyc_file).expect("Invalid pyc file");
}
