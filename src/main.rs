use std::io;

use rusty_markov::MarkovGenerator;


fn main() {
    read_stdin_lines();
}

/// Reads lines from stdin
pub fn read_stdin_lines() {
    let stdin = io::stdin().lock();

    let mut mark = MarkovGenerator::new();
    mark.train(stdin);

    println!("{}", mark.take(100).collect::<Vec<_>>().join(" "));
}