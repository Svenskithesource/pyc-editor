/// The amount of extended_args necessary to represent the arg.
/// This is more efficient than `get_extended_args` as we only calculate the count and the actual values.
pub fn get_extended_args_count(arg: u32) -> u8 {
    if arg <= u16::MAX.into() {
        1
    } else if arg <= 0xffffff {
        2
    } else {
        3
    }
}
