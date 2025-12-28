//! train module
//!
//! Contains logic for training the transitions for token prediction
use std::collections::VecDeque;
use std::io::BufRead;

use crate::tokenizer::Tokenizer;
use crate::transitions::Transitions;
use crate::BoundaryConfigs;


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
pub fn train<'a, R: BufRead>(
    input: R,
    transitions: &'a mut Transitions,
    boundary_config: &BoundaryConfigs,
    order: usize
) -> &'a mut Transitions {
    let tokenizer = Tokenizer::new(
        input,
        boundary_config.clone(),
    );

    let mut tokens = VecDeque::new();
    for next_token in tokenizer {
        let token_length = tokens.len();

        if token_length > 0 {
            transitions.count_transition(&tokens, &next_token);
            // Build up to `order` length, then maintain that till we're done
            if token_length >= order {  // '>=' instead of '>' here because we add 1 later
                tokens.pop_front();
            }
        }

        tokens.push_back(next_token);

    }

    transitions
}


#[cfg(test)]
mod tests {
    use std::{collections::HashMap, io::Cursor};
    use crate::token::Token;
    use crate::transitions::Transitions;
    use crate::BoundaryConfigs;
    use crate::train::train;


    #[test]
    fn test_train_something_simple() {
        let input = Cursor::new("
        One two three.
        ");

        let mut transitions = Transitions::new();
        train(
            input,
            &mut transitions,
            &BoundaryConfigs::LineEndings,
            2
        );

        assert_eq!(
            Transitions::with_data(
                HashMap::from([
                    (vec![Token::Boundary(String::from(""))], HashMap::from([(Token::from("One"), 1)])),
                    (vec![Token::from("One")], HashMap::from([(Token::from("two"), 1)])),
                    (vec![Token::Boundary(String::from("")), Token::from("One")], HashMap::from([(Token::from("two"), 1)])),
                    (vec![Token::from("two")], HashMap::from([(Token::from("three"), 1)])),
                    (vec![Token::from("One"), Token::from("two")], HashMap::from([(Token::from("three"), 1)])),
                    (vec![Token::from("three")], HashMap::from([(Token::from("."), 1)])),
                    (vec![Token::from("two"), Token::from("three")], HashMap::from([(Token::from("."), 1)])),
                    (vec![Token::from(".")], HashMap::from([(Token::Boundary(String::from("\n")), 1)])),
                    (vec![Token::from("three"), Token::from(".")], HashMap::from([(Token::Boundary(String::from("\n")), 1)])),
                ])
            ),
            transitions,
        );
    }

    #[test]
    fn test_tokenize_song_line_endings() {

        let input = Cursor::new("
        I see a little silhouetto of a man.
        Scaramouche, Scaramouche, will you do the Fandango?
        ");

        let mut transitions = Transitions::new();
        train(input, &mut transitions, &BoundaryConfigs::LineEndings, 2);

        assert_eq!(
            transitions,
            Transitions::with_data(
                HashMap::from([
                    (vec![Token::Boundary(String::from(""))], HashMap::from([(Token::from("I"), 1), (Token::from("Scaramouche"), 1)])),
                    (vec![Token::from("I")], HashMap::from([(Token::from("see"), 1)])),
                    (vec![Token::from("see")], HashMap::from([(Token::from("a"), 1)])),
                    (vec![Token::from("a")], HashMap::from([(Token::from("little"), 1), (Token::from("man"), 1)])),
                    (vec![Token::from("little")], HashMap::from([(Token::from("silhouetto"), 1)])),
                    (vec![Token::from("silhouetto")], HashMap::from([(Token::from("of"), 1)])),
                    (vec![Token::from("of")], HashMap::from([(Token::from("a"), 1)])),
                    (vec![Token::from("man")], HashMap::from([(Token::from("."), 1)])),
                    (vec![Token::from(".")], HashMap::from([(Token::Boundary(String::from("\n")), 1)])),
                    (vec![Token::Boundary(String::from("\n"))], HashMap::from([(Token::from("Scaramouche"), 1)])),
                    (vec![Token::from("Scaramouche")], HashMap::from([(Token::from(","), 2)])),
                    (vec![Token::from(",")], HashMap::from([(Token::from("Scaramouche"), 1), (Token::from("will"), 1)])),
                    (vec![Token::from("will")], HashMap::from([(Token::from("you"), 1)])),
                    (vec![Token::from("you")], HashMap::from([(Token::from("do"), 1)])),
                    (vec![Token::from("do")], HashMap::from([(Token::from("the"), 1)])),
                    (vec![Token::from("the")], HashMap::from([(Token::from("Fandango"), 1)])),
                    (vec![Token::from("Fandango")], HashMap::from([(Token::from("?"), 1)])),
                    (vec![Token::from("?")], HashMap::from([(Token::Boundary(String::from("\n")), 1)])),
                    (vec![Token::Boundary(String::from("")), Token::from("I")], HashMap::from([(Token::from("see"), 1)])),
                    (vec![Token::from("I"), Token::from("see")], HashMap::from([(Token::from("a"), 1)])),
                    (vec![Token::from("see"), Token::from("a")], HashMap::from([(Token::from("little"), 1)])),
                    (vec![Token::from("a"), Token::from("little")], HashMap::from([(Token::from("silhouetto"), 1)])),
                    (vec![Token::from("little"), Token::from("silhouetto")], HashMap::from([(Token::from("of"), 1)])),
                    (vec![Token::from("silhouetto"), Token::from("of")], HashMap::from([(Token::from("a"), 1)])),
                    (vec![Token::from("of"), Token::from("a")], HashMap::from([(Token::from("man"), 1)])),
                    (vec![Token::from("a"), Token::from("man")], HashMap::from([(Token::from("."), 1)])),
                    (vec![Token::from("man"), Token::from(".")], HashMap::from([(Token::Boundary(String::from("\n")), 1)])),
                    (vec![Token::from("."), Token::Boundary(String::from("\n"))], HashMap::from([(Token::from("Scaramouche"), 1)])),
                    (vec![Token::Boundary(String::from("\n")), Token::from("Scaramouche")], HashMap::from([(Token::from(","), 1)])),
                    (vec![Token::from("Scaramouche"), Token::from(",")], HashMap::from([(Token::from("Scaramouche"), 1), (Token::from("will"), 1)])),
                    (vec![Token::from(","), Token::from("Scaramouche")], HashMap::from([(Token::from(","), 1)])),
                    (vec![Token::from(","), Token::from("will")], HashMap::from([(Token::from("you"), 1)])),
                    (vec![Token::from("will"), Token::from("you")], HashMap::from([(Token::from("do"), 1)])),
                    (vec![Token::from("you"), Token::from("do")], HashMap::from([(Token::from("the"), 1)])),
                    (vec![Token::from("do"), Token::from("the")], HashMap::from([(Token::from("Fandango"), 1)])),
                    (vec![Token::from("the"), Token::from("Fandango")], HashMap::from([(Token::from("?"), 1)])),
                    (vec![Token::from("Fandango"), Token::from("?")], HashMap::from([(Token::Boundary(String::from("\n")), 1)])),
                ]),
            )
        )
    }

    #[test]
    fn test_tokenize_song_sentence_endings() {

        let input = Cursor::new("
        I see a little silhouetto of a man.
        Scaramouche, Scaramouche, will you do the Fandango?
        ");

        let mut transitions = Transitions::new();
        train(input, &mut transitions, &BoundaryConfigs::SentenceEndings, 2);

        assert_eq!(
            transitions,
            Transitions::with_data(
                HashMap::from([
                    (vec![Token::Boundary(String::from(""))], HashMap::from([(Token::from("I"), 1), (Token::from("Scaramouche"), 1)])),
                    (vec![Token::from("I")], HashMap::from([(Token::from("see"), 1)])),
                    (vec![Token::from("see")], HashMap::from([(Token::from("a"), 1)])),
                    (vec![Token::from("a")], HashMap::from([(Token::from("little"), 1), (Token::from("man"), 1)])),
                    (vec![Token::from("silhouetto")], HashMap::from([(Token::from("of"), 1)])),
                    (vec![Token::from("of")], HashMap::from([(Token::from("a"), 1)])),
                    (vec![Token::from("little")], HashMap::from([(Token::from("silhouetto"), 1)])),
                    (vec![Token::from("man")], HashMap::from([(Token::Boundary(String::from(".")), 1)])),
                    (vec![Token::Boundary(String::from("."))], HashMap::from([(Token::from("Scaramouche"), 1)])),
                    (vec![Token::from("Scaramouche")], HashMap::from([(Token::from(","), 2)])),
                    (vec![Token::from(",")], HashMap::from([(Token::from("Scaramouche"), 1), (Token::from("will"), 1)])),
                    (vec![Token::from("will")], HashMap::from([(Token::from("you"), 1)])),
                    (vec![Token::from("you")], HashMap::from([(Token::from("do"), 1)])),
                    (vec![Token::from("do")], HashMap::from([(Token::from("the"), 1)])),
                    (vec![Token::from("the")], HashMap::from([(Token::from("Fandango"), 1)])),
                    (vec![Token::from("Fandango")], HashMap::from([(Token::Boundary(String::from("?")), 1)])),
                    (vec![Token::Boundary(String::from("")), Token::from("I")], HashMap::from([(Token::from("see"), 1)])),
                    (vec![Token::from("I"), Token::from("see")], HashMap::from([(Token::from("a"), 1)])),
                    (vec![Token::from("see"), Token::from("a")], HashMap::from([(Token::from("little"), 1)])),
                    (vec![Token::from("a"), Token::from("little")], HashMap::from([(Token::from("silhouetto"), 1)])),
                    (vec![Token::from("little"), Token::from("silhouetto")], HashMap::from([(Token::from("of"), 1)])),
                    (vec![Token::from("silhouetto"), Token::from("of")], HashMap::from([(Token::from("a"), 1)])),
                    (vec![Token::from("of"), Token::from("a")], HashMap::from([(Token::from("man"), 1)])),
                    (vec![Token::from("a"), Token::from("man")], HashMap::from([(Token::Boundary(String::from(".")), 1)])),
                    (vec![Token::from("man"), Token::Boundary(String::from("."))], HashMap::from([(Token::from("Scaramouche"), 1)])),
                    (vec![Token::Boundary(String::from(".")), Token::from("Scaramouche")], HashMap::from([(Token::from(","), 1)])),
                    (vec![Token::from("Scaramouche"), Token::from(",")], HashMap::from([(Token::from("Scaramouche"), 1), (Token::from("will"), 1)])),
                    (vec![Token::from(","), Token::from("Scaramouche")], HashMap::from([(Token::from(","), 1)])),
                    (vec![Token::from(","), Token::from("will")], HashMap::from([(Token::from("you"), 1)])),
                    (vec![Token::from("will"), Token::from("you")], HashMap::from([(Token::from("do"), 1)])),
                    (vec![Token::from("you"), Token::from("do")], HashMap::from([(Token::from("the"), 1)])),
                    (vec![Token::from("do"), Token::from("the")], HashMap::from([(Token::from("Fandango"), 1)])),
                    (vec![Token::from("the"), Token::from("Fandango")], HashMap::from([(Token::Boundary(String::from("?")), 1)])),
                ])
            )
        )
    }
}