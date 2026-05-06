use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct HandleFactory {
    counters: HashMap<&'static str, u64>,
}

impl HandleFactory {
    pub fn next(&mut self, prefix: &'static str) -> String {
        let counter = self.counters.entry(prefix).or_insert(0);
        *counter += 1;
        format!("{prefix}:{}", *counter)
    }
}
