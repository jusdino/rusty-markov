use std::io;

use clap::Parser;
use rusty_markov::{Args, MarkovGenerator, BoundaryConfigs};

fn main() {
    let args = Args::parse();
    read_stdin_lines(args.max_tokens, args.boundaries);
}

/// Reads lines from stdin
pub fn read_stdin_lines(count: usize, boundary_config: BoundaryConfigs) {
    let stdin = io::stdin().lock();

    let mut mark = MarkovGenerator::new(boundary_config);
    mark.train(stdin);

    println!("{}", mark.take(count).collect::<Vec<_>>().join(" "));
}