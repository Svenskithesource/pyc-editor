use criterion::{criterion_group, criterion_main, Criterion};
use pyc_editor::v310::{
    ext_instructions::ExtInstructions,
    instructions::{Instruction, Instructions},
    opcodes::Opcode,
};
use std::hint::black_box;

fn resolve_instructions(instructions: Instructions) -> ExtInstructions {
    instructions.to_resolved()
}

fn to_instructions(instructions: ExtInstructions) -> Instructions {
    instructions.to_instructions()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut instructions = Instructions::with_capacity(200_000);

    for index in 0..100_000 {
        if index <= u8::MAX.into() {
            instructions.append_instruction(Instruction::JumpAbsolute(index as u8));
        } else {
            let mut ext_args = Vec::new();
            let mut remaining = index >> 8;
            while remaining > 0 {
                ext_args.push((remaining & 0xff) as u8);
                remaining >>= 8;
            }

            // Emit EXTENDED_ARGs in reverse order (most significant first)
            for &ext in ext_args.iter().rev() {
                instructions.append_instruction((Opcode::EXTENDED_ARG, ext).into());
            }

            instructions.append_instruction(Instruction::JumpAbsolute((index & 0xff) as u8));
        }
    }

    let resolved_instructions = instructions.to_resolved();

    c.bench_function("resolve extended_arg 100K", |b| {
        b.iter(|| resolve_instructions(black_box(instructions.clone())))
    });

    c.bench_function("extended_arg dump 100K", |b| {
        b.iter(|| to_instructions(black_box(resolved_instructions.clone())))
    });

    let mut instructions = Instructions::with_capacity(10);

    for index in 0..20 {
        if index <= u8::MAX.into() {
            instructions.append_instruction(Instruction::JumpAbsolute(index as u8));
        } else {
            let mut ext_args = Vec::new();
            let mut remaining = index >> 8;
            while remaining > 0 {
                ext_args.push((remaining & 0xff) as u8);
                remaining >>= 8;
            }

            // Emit EXTENDED_ARGs in reverse order (most significant first)
            for &ext in ext_args.iter().rev() {
                instructions.append_instruction((Opcode::EXTENDED_ARG, ext).into());
            }

            instructions.append_instruction(Instruction::JumpAbsolute((index & 0xff) as u8));
        }
    }

    let resolved_instructions = instructions.to_resolved();

    c.bench_function("resolve extended_arg 10", |b| {
        b.iter(|| resolve_instructions(black_box(instructions.clone())))
    });

    c.bench_function("extended_arg dump 10", |b| {
        b.iter(|| to_instructions(black_box(resolved_instructions.clone())))
    });
}

criterion_group! {
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().significance_level(0.05).sample_size(50);
    targets = criterion_benchmark
}
criterion_main!(benches);
