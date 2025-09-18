use std::collections::HashMap;
use std::env;
use std::io;

use rusty_markov::train::train_with_stream;
use rusty_markov::babble::babble;


fn main() {
    let args = env::args();

    for arg in args {
        println!("Received arg: {arg}");
    }

    read_stdin_lines();
}

/// Reads lines from stdin
pub fn read_stdin_lines() {
    let stdin = io::stdin().lock();

    let mut probability = HashMap::new();

    probability = train_with_stream(stdin, probability);

    eprintln!("{} n-grams trained with corpus", probability.iter().count());

    babble(&probability, 30);
}