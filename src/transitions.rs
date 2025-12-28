use std::collections::HashMap;
use crate::{as_context::AsContext, token::Token};


/// Token transitions training container
/// Counts transitions between tokens for a training corpus
#[derive(Eq, Debug)]
pub struct Transitions {
    transitions: HashMap<Vec<Token>, HashMap<Token, u32>>,
}

impl PartialEq for Transitions {
    fn eq(&self, other: &Self) -> bool {
        self.transitions == other.transitions
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
            transitions: HashMap::new(),
        }
    }

    #[cfg(test)]
    pub fn with_data(
        transitions: HashMap<Vec<Token>, HashMap<Token, u32>>,
    ) -> Self {
        Self {
            transitions,
        }
    }

    /// Add context->next_token to the transitions count training data
    /// 
    /// Will iterate over each slice of context tokens, adding transitions for each order. For example:
    /// context = vec!["four", "three", "two", "one"]
    /// next_token = "next"
    /// Would count each of these transitions:
    /// ["four", "three", "two", "one"] -> "next" (4th order)
    /// ["three", "two", "one"] -> "next"         (3rd order)
    /// ["two", "one"] -> "next"                  (2nd order)
    /// ["one"] -> "next"                         (1st order)
    pub fn count_transition<C: AsContext>(&mut self, context: C, next_token: &Token) {
        let context_vec = context.as_context();

        // Special handling of boundaries - to give us lots of initial first-order transitions, we will associate
        // every transition with a Boundary(_) as a transition from our starting token, Boundary("").
        // Only apply this for first-order contexts (single token) that are non-empty boundaries
        if context_vec.len() == 1 {
            if let Token::Boundary(val) = &context_vec[0] {
                if val != "" {
                    self.count_transition(&Token::Boundary(String::from("")), next_token);
                }
            }
        } else if context_vec.len() > 1 {
            // First, count transitions for the next smaller order
            self.count_transition(&context_vec[1..], next_token);
        }

        // Get collected transitions from context
        let token_trans = self.transitions
            .entry(context_vec)
            .or_insert_with(HashMap::new);

        // Add 1 to the transition to next_token
        token_trans.entry(next_token.clone())
            .and_modify(|p| { *p += 1 })
            .or_insert(1);

    }

    /// Get next token transition counts
    pub fn transitions_for_context<C: AsContext>(&self, context: C) -> Option<&HashMap<Token, u32>> {
        let context = context.as_context();
        if context.len() > 0 {
            let trans_for_ctx = match self.transitions.get(&context) {
                Some(t) => Some(t),
                // If no match, try next lower order context
                None => {
                    eprintln!("Failed to match {:?} context", context);
                    self.transitions_for_context(&context[1..])
                }
            };

            return trans_for_ctx
        }

        None
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_transitions_counts_first_order() {
        let mut transitions = Transitions::new();
        let last_token = Token::from("last");
        let next_token = Token::from("next");

        transitions.count_transition(&last_token, &next_token);

        assert_eq!(
            transitions,
            Transitions::with_data(
                HashMap::from([
                    (vec![last_token], HashMap::from([(next_token, 1u32)]))
                ]),
            ),
        );
    }

    #[test]
    fn test_new_transictions_counts_fourth_order() {
        let mut transitions = Transitions::new();
        let context = vec![Token::from("four"), Token::from("three"), Token::from("two"), Token::from("one")];
        let next_token = Token::from("next");

        transitions.count_transition(&context, &next_token);

        assert_eq!(
            transitions,
            Transitions::with_data(
                HashMap::from([
                    (vec![Token::from("one")], HashMap::from([(Token::from("next"), 1u32)])),
                    (vec![Token::from("two"), Token::from("one")], HashMap::from([(Token::from("next"), 1u32)])),
                    (vec![Token::from("three"), Token::from("two"), Token::from("one")], HashMap::from([(Token::from("next"), 1u32)])),
                    (vec![Token::from("four"), Token::from("three"), Token::from("two"), Token::from("one")], HashMap::from([(Token::from("next"), 1u32)])),
                ]),
            ),
        );
    }

    #[test]
    fn test_new_transitions_is_empty() {
        let transitions = Transitions::new();

        assert_eq!(
            transitions,
            Transitions::with_data(
                HashMap::new(),
            )
        );
    }

    #[test]
    fn test_transitions_equal() {
        let transitions_left = Transitions::with_data(
            HashMap::from([
                (vec![Token::from("a")], HashMap::from([(Token::from("1"), 1u32)])),
                (vec![Token::from("b"), Token::from("c")], HashMap::from([(Token::from("2"), 2u32)]))
            ]),
        );

        let transitions_right = Transitions::with_data(
            HashMap::from([
                (vec![Token::from("a")], HashMap::from([(Token::from("1"), 1u32)])),
                (vec![Token::from("b"), Token::from("c")], HashMap::from([(Token::from("2"), 2u32)]))
            ])
        );

        assert_eq!(transitions_left, transitions_right)
    }

    #[test]
    fn test_transitions_not_equal() {
        let transitions_left = Transitions::with_data(
            HashMap::from([
                (vec![Token::from("a")], HashMap::from([(Token::from("1"), 1u32)])),
                (vec![Token::from("b"), Token::from("c")], HashMap::from([(Token::from("2"), 2u32)]))
            ])
        );

        let transitions_right = Transitions::with_data(
            HashMap::from([
                (vec![Token::from("a")], HashMap::from([(Token::from("1"), 1u32)])),
                (vec![Token::from("b"), Token::from("c")], HashMap::from([(Token::from("OOPS"), 2u32)]))
            ])
        );

        assert_ne!(transitions_left, transitions_right)
    }
}