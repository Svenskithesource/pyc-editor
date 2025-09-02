# pyc_editor
`pyc_editor` is a library for reading, modifying, and writing Python `.pyc` files in Rust. It can be used for disassembling bytecode or modifying it. The library will automatically modify the instructions based on what changes you make, so you can safely edit the bytecode without worrying about offsets. It is still under heavy development, so expect breaking changes in the future.

## Supported versions
This library currently supports 3.10, 3.11, 3.12, and 3.13, with more versions planned for the future.
Each version has its own feature, so to use a version make sure to enable this feature.

## Installation
Use `cargo add pyc_editor` to add this library to your project.

## Usage
Check out the [documentation](https://docs.rs/pyc_editor) for more information.
There are examples available in the `examples` directory.

## Testing
This library is very thoroughly tested. To ensure it can output the exact same bytes as the input data, we rewrite the whole standard library and compare the output with the input. It produces a 1:1 copy of the input data.
You can run the tests with `cargo test` (integration tests only work on Windows and GitHub actions).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License
This project is licensed under the GNU GPL v3.0 license. See `LICENSE` for more information.
