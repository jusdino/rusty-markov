use std::collections::VecDeque;
use std::{collections::HashMap, io::BufRead};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

use crate::token::Token;
use crate::train::train;
use crate::transitions::Transitions;
use crate::BoundaryConfigs;


pub struct MarkovGenerator{
    order: usize,
    boundary_config: BoundaryConfigs,
    transitions: Transitions,
    rng: rand::rngs::ThreadRng,
    context: VecDeque<Token>,
}

/// Generates text, based on its traniing data, following a "markov chain" process
///
/// # Examples
/// ```rust
/// use std::io::Cursor;
/// use rusty_markov::{MarkovGenerator, BoundaryConfigs};
///
/// let mut generator = MarkovGenerator::new(BoundaryConfigs::LineEndings, 2);
/// // This should force a predictable generation loop, since there is only one transition available
/// // to each token
/// let input = Cursor::new("start middle end");
/// generator.train(input);
///
/// // Collect 5 tokens
/// let tokens: Vec<String> = generator.take(5).collect();
///
/// // Should be able to generate a chain
/// // ["start", "middle", "end", "\n"]
/// assert_eq!(tokens.len(), 4, "Should generate 3 tokens");
/// ```
impl MarkovGenerator {
    pub fn new(boundary_config: BoundaryConfigs, order: usize) -> Self {
        Self {
            // TODO: Wire this into a cli-argument
            order,
            boundary_config,
            transitions: Transitions::new(),
            rng: rand::rng(),
            context: VecDeque::from([Token::Boundary(String::from(""))]),
        }
    }

    pub fn train<R: BufRead>(&mut self, input: R) {
        train(input, &mut self.transitions, &self.boundary_config, self.order);
    }

    fn pick_next_token(&mut self) -> Option<&Token> {
        let trans_for_ctx = match self.transitions.transitions_for_context(&self.context) {
            Some(p) => p,
            None => {
                // If nothing is not in our token_transitions, stop now
                eprintln!("Failed to match any part of the context!");
                self.context.push_back(Token::Boundary(String::from("\n")));
                return None;
            }
        };

        let (counts, tokens) = decompose_transitions(trans_for_ctx);

        eprintln!("Making random choice from {} options", counts.len());
        let dist = match WeightedIndex::new(counts) {
            Ok(dist) => dist,
            Err(e) => {
                // This could happen if weights are empty, all zero, or other invalid conditions
                eprintln!("Warning: Failed to create weighted distribution: {:?}", e);
                return None;
            }
        };
        let next_token = tokens[dist.sample(&mut self.rng)];
        self.context.push_back(next_token.clone());

        // Keep context window at order length, max
        if self.context.len() > self.order {
            self.context.pop_front();
        }

        Some(next_token)
    }
}

impl Iterator for MarkovGenerator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        eprintln!("Next item with context: {:?}", self.context);

        // First check, if we've already returned a Boundary (should never return None, so we unwrap)
        if let Token::Boundary(value) = &self.context.back().unwrap() {
            // Initial state is a special Boundary("") - ignore that case
            eprintln!("last_token is a Boundary token");
            if value != "" {
                eprintln!("last_token Boundary is not empty");
                return None
            }
        }

        let next_token = match self.pick_next_token() {
            Some(token) => token.clone(),
            None => Token::Boundary(String::from("\n"))
        };

        // Wrap up a new Token for moving out
        match next_token {
            Token::Token(value) => Some(value),
            // self.last_token is now a Boundary, so next iteration will return None
            Token::Boundary(value) => Some(value),
        }
    }
}

/// Decompose next_token transitions into a pair of arrays, ready for use in the rand lib
fn decompose_transitions(trans_map: &HashMap<Token, u32>) -> (Vec<u32>, Vec<&Token>) {
    let mut counts= Vec::new();
    let mut tokens = Vec::new();

    for (k, v) in trans_map.iter() {
        tokens.push(k);
        counts.push(*v);
    }

    (counts, tokens)
}


#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use super::*;

    #[test]
    fn test_generator_properties_chain() {
        let mut generator = MarkovGenerator::new(BoundaryConfigs::LineEndings, 2);
        // This should force a predictable generation loop, since there is only one transition available
        // to each token
        let input = Cursor::new("1 2 3 4 5 6");
        generator.train(input);

        // Collect 5 tokens
        let tokens: Vec<String> = generator.take(5).collect();

        assert_eq!(
            vec!["1", "2", "3", "4", "5"],
            tokens,
        );
    }

    #[test]
    fn test_generator_empty_training() {
        let generator = MarkovGenerator::new(BoundaryConfigs::LineEndings, 2);
        // No training data

        // Should return None immediately
        eprintln!("Running generator with no training data");
        let tokens: Vec<String> = generator.collect();
        assert_eq!(
            vec!["\n"],
            tokens
        )
    }

    #[test]
    fn test_generator_dead_end_token() {
        let mut generator = MarkovGenerator::new(BoundaryConfigs::LineEndings, 2);
        let input = Cursor::new("start deadend");
        generator.train(input);

        // Should generate start, then deadend, then stop
        let tokens: Vec<String> = generator.take(10).collect();

        assert_eq!(
            vec!["start", "deadend", "\n"],
            tokens
        );
    }
}