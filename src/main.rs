use std::io;

use clap::Parser;
use rusty_markov::{Args, MarkovGenerator, BoundaryConfigs};

fn main() {
    let args = Args::parse();
    read_and_generate(args.max_tokens, args.boundaries, args.order);
}

/// Reads lines from stdin, generate some text
pub fn read_and_generate(count: usize, boundary_config: BoundaryConfigs, order: usize) {
    let stdin = io::stdin().lock();

    let mut mark = MarkovGenerator::new(boundary_config, order);
    mark.train(stdin);

    println!("{}", mark.take(count).collect::<Vec<_>>().join(" "));
}
