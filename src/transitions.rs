use std::{collections::HashMap};
use crate::token::Token;


/// Token transitions training container
/// Counts transitions between tokens for a training corpus
#[derive(Eq, Debug)]
pub struct Transitions {
    transitions: HashMap<Token, HashMap<Token, u32>>,
}

/// Allows equality comparison to a raw HashMap container, for easier testing
impl PartialEq for Transitions {
    fn eq(&self, other: &Self) -> bool {
        self.transitions == other.transitions
    }
}

impl PartialEq<HashMap<Token, HashMap<Token, u32>>> for Transitions {
    fn eq(&self, other: &HashMap<Token, HashMap<Token, u32>>) -> bool {
        self.transitions == *other
    }
}

impl PartialEq<Transitions> for HashMap<Token, HashMap<Token, u32>> {
    fn eq(&self, other: &Transitions) -> bool {
        *self == other.transitions
    }
}

#[cfg(feature = "memory-profiling")]
use memuse::DynamicUsage;

#[cfg(feature = "memory-profiling")]
impl DynamicUsage for Transitions {
    fn dynamic_usage(&self) -> usize {
        self.transitions.dynamic_usage()
    }
    
    fn dynamic_usage_bounds(&self) -> (usize, Option<usize>) {
        self.transitions.dynamic_usage_bounds()
    }
}

impl Transitions {
    /// Construct a new, empty Transitions container
    pub fn new() -> Transitions {
        Transitions {
            transitions: HashMap::new()
        }
    }

    /// Add the last_token to next_token to the transitions count training data
    pub fn count_transition(&mut self, last_token: &Token, next_token: &Token) {
        // Get collected transitions from last_token
        let token_trans = self.transitions
            .entry(last_token.clone())
            .or_insert_with(HashMap::new);

        // Add 1 to the transition to next_token
        token_trans.entry(next_token.clone())
            .and_modify(|p| { *p += 1 })
            .or_insert(1);
    }

    /// Retrieve all last_tokens as an iterator
    pub fn last_tokens(&self) -> impl Iterator<Item = &Token> {
        self.transitions.keys()
    }

    /// Get next token transition counts
    pub fn next_tokens(&self, last_token: &Token) -> Option<&HashMap<Token, u32>> {
        self.transitions.get(last_token)
    }

    /// Get the Start transition counts
    pub fn start_tokens(&self) -> Option<&HashMap<Token, u32>> {
        self.transitions.get(&Token::Terminal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_transitions_counts() {
        let mut transitions = Transitions::new();
        let last_token = Token::from("last");
        let next_token = Token::from("next");

        transitions.count_transition(&last_token, &next_token);

        assert_eq!(
            transitions,
            HashMap::from([
                (last_token, HashMap::from([(next_token, 1u32)]))
            ]),
        );
    }

    #[test]
    fn test_new_transitions_is_empty() {
        let transitions = Transitions::new();

        assert_eq!(
            transitions,
            HashMap::new(),
        );
    }
}