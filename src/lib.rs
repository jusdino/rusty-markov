mod generator;
mod token;
mod tokenize;
mod train;
mod transitions;


use clap::{command, Parser};
pub use generator::MarkovGenerator;


#[derive(Debug, Clone, PartialEq, clap::ValueEnum)]
pub enum BoundaryConfigs {
    /// Line endings are boundaries (like in a play transcript)
    LineEndings,
    /// Sentence endings are boundaries (like most anything else)
    SentenceEndings,
}

/// A Markov chain text generator
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Number of tokens to generate
    #[arg(short, long, default_value_t = 100)]
    pub max_tokens: usize,

    /// Boundary configuration for training
    #[arg(short, long, value_enum, default_value = "line-endings")]
    pub boundaries: BoundaryConfigs,
}
