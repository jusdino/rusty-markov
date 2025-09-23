//! train module
//!
//! Contains logic for training the transitions for token prediction

use std::collections::HashMap;
use std::io::BufRead;

use crate::tokenize::tokenize;

/// Read lines from buffer and train on token transitions
pub fn train_with_stream<R: BufRead>(
    input: R, transitions: &mut HashMap<String, HashMap<String, u32>>
) -> &mut HashMap<String, HashMap<String, u32>> {

    // We don't really care about breaking this up into lines, but going lower-level would mean
    // messing with reading raw bytes out of the buffer, just to reconstruct them back into utf-8
    // which would be tedious and inefficient.
    // Instead, we'll read strings out of the buffer, line-by-line, then stitch the end of one
    // line to the beginning
    let mut last_token: Option<String> = None;
    for line_res in input.lines() {
        let mut tokens: Vec<String> = match &last_token {
            // Prepend the last token from the last line, to preserve the transition
            // that would otherwise be lost across the line break
            Some(t) => Vec::from([String::from(t)]),
            None => Vec::new(),
        };

        match line_res {
            Ok(line) => {
                tokens.extend(tokenize(&line));

                // Save the last token for the next line
                let tokens_len = tokens.len();
                if tokens_len > 0 {
                    last_token = match tokens.get(tokens.len()-1) {
                        Some(t) => Some(t.clone()),
                        None => None
                    };
                }
            },
            Err(e) => {
                eprintln!("Error reading line: {}", e);
            }
        }

        train_with_tokens(tokens, transitions);
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
    tokens: Vec<String>, transitions: &mut HashMap<String, HashMap<String, u32>>
) -> &mut HashMap<String, HashMap<String, u32>> {
    let mut tokens_iter = tokens.iter();

    // Get the first token
    let mut last_token = match tokens_iter.next() {
        Some(token) => String::from(token),
        // If we don't get any tokens, there's no transition to add
        None => return transitions
    };

    for next_token in tokens_iter {
        // Get collected transitions from last_token
        let token_trans = transitions
            .entry(last_token)
            .or_insert_with(HashMap::new);

        // Add 1 to the transition to next_token
        token_trans.entry(String::from(next_token))
            .and_modify(|p| { *p += 1 })
            .or_insert(1);

        // Shift next_to last for next iteration
        last_token = String::from(next_token);
    }

    transitions
}


#[cfg(test)]
mod tests {
    use std::{collections::HashMap, io::Cursor};
    use super::*;


    #[test]
    fn test_tokenize_song() {

        let input = Cursor::new("
        I see a little silhouetto of a man.
        Scaramouche, Scaramouche, will you do the Fandango?
        Thunderbolt and lightning, very, very frightening me.
        (Galileo) Galileo, (Galileo) Galileo, Galileo Figaro, magnifico.
        But I'm just a poor boy, nobody loves me.
        He's just a poor boy from a poor family.
        Spare him his life from this monstrosity.
        ");

        let mut transitions = HashMap::new();
        train_with_stream(input, &mut transitions);

        assert_eq!(
            transitions,
            HashMap::from([
            (String::from("(Galileo)"), HashMap::from([(String::from("Galileo,"), 2)])),
            (String::from("But"), HashMap::from([(String::from("I'm"), 1)])),
            (String::from("Fandango?"), HashMap::from([(String::from("Thunderbolt"), 1)])),
            (String::from("Figaro,"), HashMap::from([(String::from("magnifico."), 1)])),
            (String::from("Galileo"), HashMap::from([(String::from("Figaro,"), 1)])),
            (String::from("Galileo,"), HashMap::from([(String::from("(Galileo)"), 1), (String::from("Galileo"), 1)])),
            (String::from("He's"), HashMap::from([(String::from("just"), 1)])),
            (String::from("I"), HashMap::from([(String::from("see"), 1)])),
            (String::from("I'm"), HashMap::from([(String::from("just"), 1)])),
            (String::from("Scaramouche,"), HashMap::from([(String::from("Scaramouche,"), 1), (String::from("will"), 1)])),
            (String::from("Spare"), HashMap::from([(String::from("him"), 1)])),
            (String::from("Thunderbolt"), HashMap::from([(String::from("and"), 1)])),
            (String::from("a"), HashMap::from([(String::from("little"), 1), (String::from("man."), 1), (String::from("poor"), 3)])),
            (String::from("and"), HashMap::from([(String::from("lightning,"), 1)])),
            (String::from("boy"), HashMap::from([(String::from("from"), 1)])),
            (String::from("boy,"), HashMap::from([(String::from("nobody"), 1)])),
            (String::from("do"), HashMap::from([(String::from("the"), 1)])),
            (String::from("family."), HashMap::from([(String::from("Spare"), 1)])),
            (String::from("frightening"), HashMap::from([(String::from("me."), 1)])),
            (String::from("from"), HashMap::from([(String::from("a"), 1), (String::from("this"), 1)])),
            (String::from("him"), HashMap::from([(String::from("his"), 1)])),
            (String::from("his"), HashMap::from([(String::from("life"), 1)])),
            (String::from("just"), HashMap::from([(String::from("a"), 2)])),
            (String::from("life"), HashMap::from([(String::from("from"), 1)])),
            (String::from("lightning,"), HashMap::from([(String::from("very,"), 1)])),
            (String::from("little"), HashMap::from([(String::from("silhouetto"), 1)])),
            (String::from("loves"), HashMap::from([(String::from("me."), 1)])),
            (String::from("magnifico."), HashMap::from([(String::from("But"), 1)])),
            (String::from("man."), HashMap::from([(String::from("Scaramouche,"), 1)])),
            (String::from("me."), HashMap::from([(String::from("(Galileo)"), 1), (String::from("He's"), 1)])),
            (String::from("nobody"), HashMap::from([(String::from("loves"), 1)])),
            (String::from("of"), HashMap::from([(String::from("a"), 1)])),
            (String::from("poor"), HashMap::from([(String::from("boy"), 1), (String::from("boy,"), 1), (String::from("family."), 1)])),
            (String::from("see"), HashMap::from([(String::from("a"), 1)])),
            (String::from("silhouetto"), HashMap::from([(String::from("of"), 1)])),
            (String::from("the"), HashMap::from([(String::from("Fandango?"), 1)])),
            (String::from("this"), HashMap::from([(String::from("monstrosity."), 1)])),
            (String::from("very"), HashMap::from([(String::from("frightening"), 1)])),
            (String::from("very,"), HashMap::from([(String::from("very"), 1)])),
            (String::from("will"), HashMap::from([(String::from("you"), 1)])),
            (String::from("you"), HashMap::from([(String::from("do"), 1)])),
            ])
        )
    }

    #[test]
    fn test_train_with_tokens_populates_transitions_map() {
        let mut transitions = HashMap::new();
        let tokens = vec![
            String::from("I"),
            String::from("see"),
            String::from("a"),
            String::from("little"),
            String::from("silhouetto"),
            String::from("of"),
            String::from("a"),
            String::from("man.")
        ];

        train_with_tokens(tokens, &mut transitions);

        assert_eq!(
            transitions,
            HashMap::from([
                (String::from("I"), HashMap::from([(String::from("see"), 1)])),
                (String::from("see"), HashMap::from([(String::from("a"), 1)])),
                (String::from("a"), HashMap::from([(String::from("little"), 1), (String::from("man."), 1)])),
                (String::from("little"), HashMap::from([(String::from("silhouetto"), 1)])),
                (String::from("silhouetto"), HashMap::from([(String::from("of"), 1)])),
                (String::from("of"), HashMap::from([(String::from("a"), 1)])),
            ])
        );
    }
}