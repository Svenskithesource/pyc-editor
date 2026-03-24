use criterion::{Criterion, criterion_group, criterion_main};
use pyc_editor::cfg::create_cfg;
use pyc_editor::utils::UnusedArgument;
use pyc_editor::v311::code_objects::RelativeJump;
use pyc_editor::v311::ext_instructions::ExtInstruction;
use std::hint::black_box;

fn generate_instructions(amount_of_blocks: usize) -> Vec<ExtInstruction> {
    let mut instructions = Vec::with_capacity((amount_of_blocks * 2) + 2);

    instructions.push(ExtInstruction::LoadConst(
        pyc_editor::v311::code_objects::ConstIndex { index: 0 },
    ));

    for index in 0..amount_of_blocks {
        instructions.push(ExtInstruction::Copy(1));
        instructions.push(ExtInstruction::PopJumpForwardIfTrue(RelativeJump {
            // Jump to RETURN
            index: ((amount_of_blocks * 2) + 2) as u32 - (index * 2 + 4) as u32,
            direction: pyc_editor::v311::code_objects::JumpDirection::Forward,
        }));
    }

    instructions.push(ExtInstruction::ReturnValue(UnusedArgument::default()));

    instructions
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("create cfg 10K blocks", |b| {
        b.iter(|| create_cfg(black_box(generate_instructions(10_000)), None))
    });

    c.bench_function("create cfg 5K blocks", |b| {
        b.iter(|| create_cfg(black_box(generate_instructions(5_000)), None))
    });

    c.bench_function("create cfg 1K blocks", |b| {
        b.iter(|| create_cfg(black_box(generate_instructions(1_000)), None))
    });

    c.bench_function("create cfg 100 blocks", |b| {
        b.iter(|| create_cfg(black_box(generate_instructions(100)), None))
    });

    c.bench_function("create cfg 25 blocks", |b| {
        b.iter(|| create_cfg(black_box(generate_instructions(25)), None))
    });
}

criterion_group! {
    name = cfg_creation;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.05).sample_size(50);
    targets = criterion_benchmark
}
criterion_main!(cfg_creation);
