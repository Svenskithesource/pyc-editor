use criterion::{Criterion, criterion_group, criterion_main};
use pyc_editor::cfg::create_cfg;
use pyc_editor::sir::cfg_to_ir;
use pyc_editor::utils::UnusedArgument;
use pyc_editor::v311::code_objects::RelativeJump;
use pyc_editor::v311::opcodes::sir::SIRNode;
use pyc_editor::v311::{
    ext_instructions::ExtInstruction,
    instructions::{Instruction, Instructions},
};
use std::hint::black_box;

fn generate_instructions(amount_of_blocks: usize) -> Vec<ExtInstruction> {
    let mut instructions = Vec::with_capacity((amount_of_blocks * 2) + 2);

    instructions.push(ExtInstruction::LoadConst(
        pyc_editor::v311::code_objects::ConstIndex { index: 0 },
    ));

    for _ in 0..amount_of_blocks {
        instructions.extend_from_slice(
            Instructions::new(vec![
                Instruction::PushNull(0),
                Instruction::LoadName(0),
                Instruction::LoadConst(0),
                Instruction::Precall(1),
                Instruction::Cache(0),
                Instruction::Call(1),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::PopTop(0),
                Instruction::LoadConst(1),
                Instruction::StoreName(1),
                Instruction::PushNull(0),
                Instruction::LoadName(0),
                Instruction::LoadConst(2),
                Instruction::LoadName(1),
                Instruction::FormatValue(2),
                Instruction::BuildString(2),
                Instruction::Precall(1),
                Instruction::Cache(0),
                Instruction::Call(1),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::Cache(0),
                Instruction::PopTop(0),
            ])
            .to_resolved()
            .unwrap()
            .as_ref(),
        );
        instructions.push(ExtInstruction::Copy(1));
        instructions.push(ExtInstruction::PopJumpForwardIfTrue(RelativeJump {
            // Jump to RETURN
            index: 0,
            direction: pyc_editor::v311::code_objects::JumpDirection::Forward,
        }));
        instructions.push(ExtInstruction::ReturnValue(UnusedArgument::default()));
    }

    instructions.push(ExtInstruction::ReturnValue(UnusedArgument::default()));

    instructions
}

fn criterion_benchmark(c: &mut Criterion) {
    let cfg = create_cfg(generate_instructions(10_000), None).unwrap();
    c.bench_function("create sir 10K blocks", |b| {
        b.iter(|| cfg_to_ir::<_, SIRNode>(black_box(&cfg), false))
    });

    let cfg = create_cfg(generate_instructions(5_000), None).unwrap();
    c.bench_function("create sir 5K blocks", |b| {
        b.iter(|| cfg_to_ir::<_, SIRNode>(black_box(&cfg), false))
    });

    let cfg = create_cfg(generate_instructions(1_000), None).unwrap();
    c.bench_function("create sir 1K blocks", |b| {
        b.iter(|| cfg_to_ir::<_, SIRNode>(black_box(&cfg), false))
    });

    let cfg = create_cfg(generate_instructions(100), None).unwrap();
    c.bench_function("create sir 100 blocks", |b| {
        b.iter(|| cfg_to_ir::<_, SIRNode>(black_box(&cfg), false))
    });

    let cfg = create_cfg(generate_instructions(25), None).unwrap();
    c.bench_function("create sir 25 blocks", |b| {
        b.iter(|| cfg_to_ir::<_, SIRNode>(black_box(&cfg), false))
    });
}

criterion_group! {
    name = sir_creation;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.05).sample_size(50);
    targets = criterion_benchmark
}
criterion_main!(sir_creation);
