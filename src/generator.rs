use std::{collections::HashMap, io::BufRead};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::seq::IteratorRandom;

use crate::train::train_with_stream;


pub struct MarkovGenerator {
    token_transitions: HashMap<String, HashMap<String, u32>>,
    rng: rand::rngs::ThreadRng,
    last_token: Option<String>,
}

/// Generates text, based on its traniing data, following a "markov chain" process
///
/// # Examples
/// ```rust
/// use std::io::Cursor;
/// use rusty_markov::MarkovGenerator;
///
/// let mut generator = MarkovGenerator::new();
/// // This should force a predictable generation loop, since there is only one transition available
/// // to each token
/// let input = Cursor::new("start middle end start middle end");
/// generator.train(input);
///
/// // Collect 5 tokens
/// let tokens: Vec<String> = generator.take(5).collect();
///
/// // Should be able to generate a chain
/// assert_eq!(tokens.len(), 5, "Should generate 5 tokens");
/// ```
impl MarkovGenerator {
    pub fn new() -> Self {
        Self {
            token_transitions: HashMap::new(),
            rng: rand::rng(),
            last_token: None,
        }
    }

    pub fn train<R: BufRead>(&mut self, input: R) {
        train_with_stream(input, &mut self.token_transitions);
    }

    fn pick_first_token(&mut self) -> Option<&String> {
        self.token_transitions
            .keys()
            // Pick a random key
            .choose(&mut self.rng)
    }

    fn pick_next_token(&mut self) -> Option<&String> {
        if let Some(last_token) = self.last_token.as_ref() {
            let next_transition_counts = match self.token_transitions.get(last_token) {
                Some(p) => p,
                None => {
                    // If last_token is not in our token_transitions, stop now
                    return None;
                }
            };

            let (counts, tokens) = decompose_transitions(next_transition_counts);

            let dist = match WeightedIndex::new(counts) {
                Ok(dist) => dist,
                Err(e) => {
                    // This could happen if weights are empty, all zero, or other invalid conditions
                    eprintln!("Warning: Failed to create weighted distribution: {:?}", e);
                    return None;
                }
            };
            let next_token = tokens[dist.sample(&mut self.rng)];

            Some(next_token)
        } else {
            // If we don't have our first token, return None
            return None;
        }

    }
}

impl Iterator for MarkovGenerator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.last_token {
            // This is the first token
            None => {
                self.last_token = self.pick_first_token().map(String::from);
            },
            // This is after the first token
            Some(_) => {
                self.last_token = self.pick_next_token().map(String::from);
            }
        };

        // Wrap up a new String for moving out
        match &self.last_token {
            None => None,
            Some(token) => Some(token.clone())
        }
    }
}

/// Decompose next_token transitions into a pair of arrays, ready for use in the rand lib
fn decompose_transitions(trans_map: &HashMap<String, u32>) -> (Vec<u32>, Vec<&String>) {
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
        let mut generator = MarkovGenerator::new();
        // This should force a predictable generation loop, since there is only one transition available
        // to each token
        let input = Cursor::new("start middle end start middle end");
        generator.train(input);

        // Collect 5 tokens
        let tokens: Vec<String> = generator.take(5).collect();

        // Should be able to generate a chain
        assert_eq!(tokens.len(), 5, "Should generate 5 tokens");

        // Each token should be one of our expected tokens
        let expected_tokens = ["start", "middle", "end"];
        for token in &tokens {
            assert!(expected_tokens.contains(&token.as_str()),
                    "Token '{}' should be one of {:?}", token, expected_tokens);
        }
    }

    #[test]
    fn test_generator_empty_training() {
        let mut generator = MarkovGenerator::new();
        // No training data

        // Should return None immediately
        let first_token = generator.next();
        assert!(first_token.is_none(), "Should return None with no training data");
    }

    #[test]
    fn test_generator_single_token_training() {
        let mut generator = MarkovGenerator::new();
        let input = Cursor::new("lonely lonely");
        generator.train(input);

        // Should generate the single token repeatedly (self-loop)
        let tokens: Vec<String> = generator.take(5).collect();
        assert_eq!(tokens.len(), 5, "Should generate 5 tokens");
        for token in &tokens {
            assert_eq!(token, "lonely", "All tokens should be 'lonely'");
        }
    }

    #[test]
    fn test_generator_dead_end_token() {
        let mut generator = MarkovGenerator::new();
        let input = Cursor::new("start deadend");
        generator.train(input);

        // Should generate start, then deadend, then stop
        let tokens: Vec<String> = generator.take(10).collect();

        assert!(tokens.len() <= 2, "Should stop at deadend token");
        assert!(tokens.len() >= 1, "Should have at least one token");

        // First token should be either "start" or "deadend" (randomly chosen)
        assert!(
            tokens[0] == "start" || tokens[0] == "deadend",
            "First token should be either 'start' or 'deadend', got: {}", tokens[0]
        );

        match tokens.len() {
            // If we have one token, the first should be "deadend"
            1 => {
                assert_eq!(tokens[0], "deadend", "First token should be deadend");
            },
            // If we have two tokens the first should be "start", second should be "deadend"
            2 => {
                assert_eq!(tokens[0], "start", "First token should be start");
                assert_eq!(tokens[1], "deadend", "Second token should be deadend");
            },
            i => panic!("tokens length should be 1 or 2, received {}", i)
        }
    }
}