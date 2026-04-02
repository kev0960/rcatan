/// A small sorted `Vec`-backed set. Deterministic iteration order, cache-friendly,
/// and faster than `BTreeSet`/`HashSet` for the tiny collections in this codebase.
pub struct VecSet<T> {
    inner: Vec<T>,
}

impl<T: Ord> VecSet<T> {
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    pub fn insert(&mut self, value: T) -> bool {
        match self.inner.binary_search(&value) {
            Ok(_) => false,
            Err(pos) => {
                self.inner.insert(pos, value);
                true
            }
        }
    }

    pub fn contains(&self, value: &T) -> bool {
        self.inner.binary_search(value).is_ok()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<T: Ord> FromIterator<T> for VecSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut inner: Vec<T> = iter.into_iter().collect();
        inner.sort_unstable();
        inner.dedup();
        Self { inner }
    }
}

impl<T> IntoIterator for VecSet<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}
