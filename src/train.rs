//! train module
//!
//! Contains logic for training the transitions for token prediction
use std::io::BufRead;

use crate::token::Token;
use crate::tokenize::tokenize;
use crate::transitions::Transitions;
use crate::BoundaryConfigs;


/// Read lines from buffer and train on token transitions
pub fn train_with_stream<'a, R: BufRead>(
    input: R, transitions: &'a mut Transitions, boundary_config: &BoundaryConfigs
) -> &'a mut Transitions {

    // We don't really care about breaking this up into lines, but going lower-level would mean
    // messing with reading raw bytes out of the buffer, just to reconstruct them back into utf-8
    // which would be tedious and inefficient.
    // Instead, we'll read strings out of the buffer, line-by-line, then stitch the end of one
    // line to the beginning of the next
    let mut last_token: Token = Token::Boundary;
    for line_res in input.lines() {
        let mut tokens: Vec<Token> = Vec::new();

        // This is the beginning of a new line so, if line-endings are our boundaries, push a Token::Boundary
        if let BoundaryConfigs::LineEndings = boundary_config {
            tokens.push(Token::Boundary);
        } else {
            // Otherwise, preserve the transition from the last line by pushing last_token
            tokens.push(last_token.clone());
        }

        match line_res {
            Ok(line) => {
                tokens.extend(tokenize(&line, boundary_config));

                if let BoundaryConfigs::SentenceEndings = boundary_config {
                    // Save the last token for the next line
                    let tokens_len = tokens.len();
                    if tokens_len > 0 {
                        last_token = match tokens.get(tokens_len-1) {
                            Some(t) => t.clone(),
                            None => Token::Boundary
                        };
                    }
                }
            },
            Err(e) => {
                eprintln!("Error reading line: {}", e);
            }
        }

        // If we're using LineEndings as boundary_config, push a Token::Boundary on the end
        if let BoundaryConfigs::LineEndings = boundary_config {
            tokens.push(Token::Boundary);
        }

        train_with_tokens(tokens, transitions);
    }

    // Log memory usage when memory-profiling feature is enabled
    #[cfg(feature = "memory-profiling")]
    {
        use memuse::DynamicUsage;
        let estimated_size = transitions.dynamic_usage();
        eprintln!(
            "Estimated transitions HashMap memory usage: {} bytes ({:.2} MB)", 
            estimated_size,
            estimated_size as f64 / 1_048_576.0
        );
    }

    transitions
}

/// Input tokens and add transitions to existing map
///
/// transitions should look like:
/// ```json
/// {
///     "the": {
///         "cat": 1,
///         "bat": 5,
///         "hat": 2,
///     },
///     "cat": {
///         "sat": 2,
///         "was": 5,
///         "ran": 1,
///     }
/// }
/// ```
pub fn train_with_tokens(
    tokens: Vec<Token>, transitions: &mut Transitions 
) -> &mut Transitions {
    let mut tokens_iter = tokens.iter();

    // Get the first token
    let mut last_token = match tokens_iter.next() {
        Some(token) => token.clone(),
        // If we don't get any tokens, there's no transition to add
        None => return transitions
    };

    for next_token in tokens_iter {
        match (&last_token, next_token) {
            // Specifically suppress Boundary->Boundary transitions caused by things like empty lines
            (Token::Boundary, Token::Boundary) => (),
            _ => transitions.count_transition(&last_token, next_token)
        };

        // Shift next to last for next iteration
        last_token = next_token.clone();
    }

    transitions
}


#[cfg(test)]
mod tests {
    use std::{collections::HashMap, io::Cursor};
    use super::*;


    #[test]
    fn test_tokenize_song_line_endings() {

        let input = Cursor::new("
        I see a little silhouetto of a man.
        Scaramouche, Scaramouche, will you do the Fandango?
        ");

        let mut transitions = Transitions::new();
        train_with_stream(input, &mut transitions, &BoundaryConfigs::LineEndings);

        assert_eq!(
            transitions,
            HashMap::from([
            (Token::Boundary, HashMap::from([(Token::from("I"), 1), (Token::from("Scaramouche"), 1)])),
            (Token::from("I"), HashMap::from([(Token::from("see"), 1)])),
            (Token::from("see"), HashMap::from([(Token::from("a"), 1)])),
            (Token::from("a"), HashMap::from([(Token::from("little"), 1), (Token::from("man"), 1)])),
            (Token::from("silhouetto"), HashMap::from([(Token::from("of"), 1)])),
            (Token::from("of"), HashMap::from([(Token::from("a"), 1)])),
            (Token::from("little"), HashMap::from([(Token::from("silhouetto"), 1)])),
            (Token::from("man"), HashMap::from([(Token::from("."), 1)])),
            (Token::from("."), HashMap::from([(Token::Boundary, 1)])),
            (Token::from("Scaramouche"), HashMap::from([(Token::from(","), 2)])),
            (Token::from(","), HashMap::from([(Token::from("Scaramouche"), 1), (Token::from("will"), 1)])),
            (Token::from("will"), HashMap::from([(Token::from("you"), 1)])),
            (Token::from("you"), HashMap::from([(Token::from("do"), 1)])),
            (Token::from("do"), HashMap::from([(Token::from("the"), 1)])),
            (Token::from("the"), HashMap::from([(Token::from("Fandango"), 1)])),
            (Token::from("Fandango"), HashMap::from([(Token::from("?"), 1)])),
            (Token::from("?"), HashMap::from([(Token::Boundary, 1)])),
            ])
        )
    }

    #[test]
    fn test_tokenize_song_sentence_endings() {

        let input = Cursor::new("
        I see a little silhouetto of a man.
        Scaramouche, Scaramouche, will you do the Fandango?
        ");

        let mut transitions = Transitions::new();
        train_with_stream(input, &mut transitions, &BoundaryConfigs::SentenceEndings);

        assert_eq!(
            transitions,
            HashMap::from([
            (Token::Boundary, HashMap::from([(Token::from("I"), 1), (Token::from("Scaramouche"), 1)])),
            (Token::from("I"), HashMap::from([(Token::from("see"), 1)])),
            (Token::from("see"), HashMap::from([(Token::from("a"), 1)])),
            (Token::from("a"), HashMap::from([(Token::from("little"), 1), (Token::from("man"), 1)])),
            (Token::from("silhouetto"), HashMap::from([(Token::from("of"), 1)])),
            (Token::from("of"), HashMap::from([(Token::from("a"), 1)])),
            (Token::from("little"), HashMap::from([(Token::from("silhouetto"), 1)])),
            (Token::from("man"), HashMap::from([(Token::Boundary, 1)])),
            (Token::from("Scaramouche"), HashMap::from([(Token::from(","), 2)])),
            (Token::from(","), HashMap::from([(Token::from("Scaramouche"), 1), (Token::from("will"), 1)])),
            (Token::from("will"), HashMap::from([(Token::from("you"), 1)])),
            (Token::from("you"), HashMap::from([(Token::from("do"), 1)])),
            (Token::from("do"), HashMap::from([(Token::from("the"), 1)])),
            (Token::from("the"), HashMap::from([(Token::from("Fandango"), 1)])),
            (Token::from("Fandango"), HashMap::from([(Token::Boundary, 1)])),
            ])
        )
    }

    #[test]
    fn test_train_with_tokens_populates_transitions_map() {
        let mut transitions = Transitions::new();
        let tokens = vec![
            Token::from("I"),
            Token::from("see"),
            Token::from("a"),
            Token::from("little"),
            Token::from("silhouetto"),
            Token::from("of"),
            Token::from("a"),
            Token::from("man.")
        ];

        train_with_tokens(tokens, &mut transitions);

        assert_eq!(
            transitions,
            HashMap::from([
                (Token::from("I"), HashMap::from([(Token::from("see"), 1)])),
                (Token::from("see"), HashMap::from([(Token::from("a"), 1)])),
                (Token::from("a"), HashMap::from([(Token::from("little"), 1), (Token::from("man."), 1)])),
                (Token::from("little"), HashMap::from([(Token::from("silhouetto"), 1)])),
                (Token::from("silhouetto"), HashMap::from([(Token::from("of"), 1)])),
                (Token::from("of"), HashMap::from([(Token::from("a"), 1)])),
            ])
        );
    }
}