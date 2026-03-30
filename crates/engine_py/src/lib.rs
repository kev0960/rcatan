#[pyo3::pymodule]
mod catan_engine {
    use pyo3::prelude::*;

    #[pyfunction]
    fn hello() -> &'static str {
        "hello from rust"
    }
}
