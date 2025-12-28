//! tokenize module
//!
//! Contains logic for tokenizing strings

use std::{collections::VecDeque, io::BufRead};
use crate::{token::Token, BoundaryConfigs};


const SENTENCE_ENDINGS: [char; 4] = ['.', '!', '?', '\n'];
const PUNCTUATION_ENDINGS: [char; 10] = ['.', '!', '?', ',', ';', '"', '”', '\'', '}', ')'];
const PUNCTUATION_BEGININGS: [char; 5] = ['"', '“', '\'', '{', '('];


pub struct Tokenizer<R: BufRead> {
    tokens: VecDeque<Token>,
    boundary_config: BoundaryConfigs,
    buf: R,
}


impl <R: BufRead>Tokenizer<R> {
    pub fn new(input: R, boundary_config: BoundaryConfigs) -> Self {
        Self {
            tokens: VecDeque::from([Token::Boundary(String::from(""))]),
            boundary_config,
            buf: input,
        }
    }

    /// Load more tokens from buf into the tokens vector
    fn load_tokens(&mut self) {
        // Keep reading lines until non-empty content is returned
        for line_res in self.buf.by_ref().lines() {
            match line_res {
                Ok(line) => {
                    let trimmed_line = line.trim();
                    if trimmed_line.len() > 0 {
                        self.tokens.extend(tokenize(&trimmed_line, &self.boundary_config));

                        // If we're using LineEndings as boundary_config, push a Token::Boundary on the end
                        if let BoundaryConfigs::LineEndings = self.boundary_config {
                            self.tokens.push_back(Token::Boundary(String::from("\n")));
                        }
                    }
                },
                Err(e) => {
                    eprintln!("Error reading line: {}", e);
                }
            }
            // We only want to read one non-empty line at a time into the tokens vector, so we'll break from the loop
            // as soon as tokens has some content
            if self.tokens.len() > 0 {
                break;
            }
        }
    }
}

impl <R: BufRead>Iterator for Tokenizer<R> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        // If we have tokens, pop one
        if let Some(val) = self.tokens.pop_front() {
            return Some(val);
        }
        
        // Otherwise, try to load more tokens
        self.load_tokens();
        
        // After loading, try to pop again (or return None if still empty)
        self.tokens.pop_front()
    }
}


/// Takes an input line of text, returns the line broken up
/// as a vector of tokens
pub fn tokenize(line: &str, boundary_config: &BoundaryConfigs) -> impl Iterator<Item = Token> {
    // Start with just splitting on whitespace
    let mut tokens: Vec<Token> = line.split_whitespace().map(|s| Token::from(s)).collect();
    if let BoundaryConfigs::SentenceEndings = boundary_config {
        split_out_sentence_boundaries(&mut tokens);
    }
    split_out_punctuation_endings(&mut tokens);
    split_out_punctuation_beginings(&mut tokens);

    tokens.into_iter()
}

/// Splits out tokens with sentence boundaries
/// `["man."]` -> `["man", Token::Boundary]`
fn split_out_sentence_boundaries(tokens: &mut Vec<Token>) {
    // Collect indices and new tokens to insert
    let mut insertions: Vec<(usize, Vec<Token>)> = Vec::new();
    
    for (i, token) in tokens.iter().enumerate() {
        if let Token::Token(value) = token {
            if let Some(last_char) = value.chars().last() {
                if SENTENCE_ENDINGS.contains(&last_char) {
                    // Create the token without the sentence ending
                    let mut new_tokens: Vec<Token> = Vec::new();
                    // If the value was only one char (i.e. ".") we'll end up adding a blank token ""
                    // so we only add the trimmed version if it's longer than 1
                    if value.len() > 1 {
                        let trimmed_value: String = value.chars().take(value.len() - 1).collect();
                        new_tokens.push(Token::Token(trimmed_value));
                    }
                    new_tokens.push(Token::Boundary(String::from(last_char)));
                    insertions.push((i, new_tokens));
                }
            }
        }
    }
    
    // Apply insertions in reverse order to maintain correct indices
    for (i, new_tokens) in insertions.into_iter().rev() {
        tokens.remove(i); // Remove the original token
        for (j, new_token) in new_tokens.into_iter().enumerate() {
            tokens.insert(i + j, new_token);
        }
    }
}


/// Splits punctuation off of the beginings/ends of words
/// `["Paren)"]` -> `["Paren", ")"]`
fn split_out_punctuation_endings(tokens: &mut Vec<Token>) {
    // Collect indices and new tokens to insert
    let mut insertions: Vec<(usize, Vec<Token>)> = Vec::new();
    
    for (i, token) in tokens.iter().enumerate() {
        if let Token::Token(value) = token {
            if let Some(last_char) = value.chars().last() {
                if PUNCTUATION_ENDINGS.contains(&last_char) {
                    let mut new_tokens: Vec<Token> = Vec::new();
                    if value.len() > 1 {
                        // Create the token without the punctuation
                        let trimmed_value: String = value.chars().take(value.len() - 1).collect();
                        new_tokens.push(Token::from(trimmed_value));
                    }
                    new_tokens.push(Token::from(last_char));
                    insertions.push((i, new_tokens));
                }
            }
        }
    }
    
    // Apply insertions in reverse order to maintain correct indices
    for (i, new_tokens) in insertions.into_iter().rev() {
        tokens.remove(i); // Remove the original token
        for (j, new_token) in new_tokens.into_iter().enumerate() {
            tokens.insert(i + j, new_token);
        }
    }
}


/// Splits punctuation off of the beginings/ends of words
/// `["(Paren"]` -> `["(", "Paren"]`
fn split_out_punctuation_beginings(tokens: &mut Vec<Token>) {
    // Collect indices and new tokens to insert
    let mut insertions: Vec<(usize, Vec<Token>)> = Vec::new();
    
    for (i, token) in tokens.iter().enumerate() {
        if let Token::Token(value) = token {
            let mut value_chars = value.chars();
            if let Some(first_char) = value_chars.next() {
                if PUNCTUATION_BEGININGS.contains(&first_char) {
                    let mut new_tokens = vec![Token::from(first_char)];
                    if value.len() > 1 {
                        // Create the token without the punctuation
                        // The first char is already iterated
                        let trimmed_value: String = value_chars.collect();
                        new_tokens.push(Token::from(trimmed_value));
                    }
                    insertions.push((i, new_tokens));
                }
            }
        }
    }
    
    // Apply insertions in reverse order to maintain correct indices
    for (i, new_tokens) in insertions.into_iter().rev() {
        tokens.remove(i); // Remove the original token
        for (j, new_token) in new_tokens.into_iter().enumerate() {
            tokens.insert(i + j, new_token);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use super::*;

    #[test]
    fn test_tokenizer_with_line_endings() {

        let input = Cursor::new("
        I see a (little) silhouetto of a man.
        Scaramouche, Scaramouche, will you do the Fandango?
        ");

        let tokenizer = Tokenizer::new(
            input,
            BoundaryConfigs::LineEndings
        );

        let tokens: Vec<Token> = tokenizer.collect();
        assert_eq!(
            tokens,
            vec![
                Token::Boundary(String::from("")),
                Token::from("I"),
                Token::from("see"),
                Token::from("a"),
                Token::from("("),
                Token::from("little"),
                Token::from(")"),
                Token::from("silhouetto"),
                Token::from("of"),
                Token::from("a"),
                Token::from("man"),
                Token::from("."),
                Token::Boundary(String::from("\n")),
                Token::from("Scaramouche"),
                Token::from(","),
                Token::from("Scaramouche"),
                Token::from(","),
                Token::from("will"),
                Token::from("you"),
                Token::from("do"),
                Token::from("the"),
                Token::from("Fandango"),
                Token::from("?"),
                Token::Boundary(String::from("\n")),
            ]
        )
    }

    #[test]
    fn test_tokenizer_with_sentence_endings() {

        let input = Cursor::new("
        I see a (little) silhouetto of a man.
        Scaramouche, Scaramouche, will you do the Fandango?
        ");

        let tokenizer = Tokenizer::new(
            input,
            BoundaryConfigs::SentenceEndings,
        );

        let tokens: Vec<Token> = tokenizer.collect();
        assert_eq!(
            tokens,
            vec![
                Token::Boundary(String::from("")),
                Token::from("I"),
                Token::from("see"),
                Token::from("a"),
                Token::from("("),
                Token::from("little"),
                Token::from(")"),
                Token::from("silhouetto"),
                Token::from("of"),
                Token::from("a"),
                Token::from("man"),
                Token::Boundary(String::from(".")),
                Token::from("Scaramouche"),
                Token::from(","),
                Token::from("Scaramouche"),
                Token::from(","),
                Token::from("will"),
                Token::from("you"),
                Token::from("do"),
                Token::from("the"),
                Token::from("Fandango"),
                Token::Boundary(String::from("?")),
            ]
        )
    }

    #[test]
    fn test_tokenize_with_sentence_endings() {
        let input = "I see a (little) silhouetto of a man.";
        let tokenized = tokenize(input, &BoundaryConfigs::SentenceEndings);
        let output: Vec<Token> = vec![
            Token::from("I"),
            Token::from("see"),
            Token::from("a"),
            Token::from("("),
            Token::from("little"),
            Token::from(")"),
            Token::from("silhouetto"),
            Token::from("of"),
            Token::from("a"),
            Token::from("man"),
            Token::Boundary(String::from(".")),
        ];

        assert_eq!(
            output,
            tokenized.collect::<Vec<Token>>(),
        )
    }

    #[test]
    fn test_tokenize_with_line_endings() {
        let input = "I see a (little) silhouetto of a man.";
        let tokenized = tokenize(input, &BoundaryConfigs::LineEndings);
        let output: Vec<Token> = vec![
            Token::from("I"),
            Token::from("see"),
            Token::from("a"),
            Token::from("("),
            Token::from("little"),
            Token::from(")"),
            Token::from("silhouetto"),
            Token::from("of"),
            Token::from("a"),
            Token::from("man"),
            Token::from("."),
        ];

        assert_eq!(
            output,
            tokenized.collect::<Vec<Token>>(),
        )
    }

    #[test]
    fn test_split_out_sentence_boundaries() {
        // Level 1: Easy
        let mut tokens = vec![Token::from("a"), Token::from("man.")];
        split_out_sentence_boundaries(&mut tokens);
        let expected: Vec<Token> = vec![
            Token::from("a"),
            Token::from("man"),
            Token::Boundary(String::from(".")),
        ];

        assert_eq!(
            expected,
            tokens,
            "Should split period off end of word"
        );

        // Level 2: Interesting - just the boundary
        let mut tokens = vec![Token::from(".")];
        split_out_sentence_boundaries(&mut tokens);
        let expected: Vec<Token> = vec![
            Token::Boundary(String::from("."))
        ];
        assert_eq!(
            expected,
            tokens,
            "Should split just a boundary without stray tokens"
        );

        // Level 3: Just weird - sentance boundaries where they don't belong
        let mut tokens = vec![
            Token::from("(something)"),
            // First sentence boundary
            Token::from("?"),
            // Second sentence boundary
            Token::from(".truly!"),
            Token::from(".odd"),
            // Third sentence boundary
            Token::from("happening."),
            Token::from("here.)"),
        ];
        split_out_sentence_boundaries(&mut tokens);
        let expected: Vec<Token> = vec![
            Token::from("(something)"),
            // First sentence boundary split
            Token::Boundary(String::from("?")),
            // Second sentence boundary split
            Token::from(".truly"),
            Token::Boundary(String::from("!")),
            Token::from(".odd"),
            // Second sentence boundary split
            Token::from("happening"),
            Token::Boundary(String::from(".")),
            Token::from("here.)"),
        ];
        assert_eq!(
            expected,
            tokens,
            "Failed level 3"
        );
    }

    #[test]
    fn test_split_out_punctuation_endings() {
        // Level 1: Easy
        let mut tokens = vec![
            Token::from("(a)"),
            Token::from("man\""),
            Token::from(")"),
        ];
        split_out_punctuation_endings(&mut tokens);
        let expected: Vec<Token> = vec![
            Token::from("(a"),
            Token::from(")"),
            Token::from("man"),
            Token::from("\""),
            Token::from(")"),
        ];

        assert_eq!(
            expected,
            tokens,
            "Should split right parens and quotes"
        );
    }

    #[test]
    fn test_split_out_punctuation_beginings() {
        // Level 1: Easy
        let mut tokens = vec![
            Token::from("(a)"),
            Token::from("\"man"),
            Token::from("("),
        ];
        split_out_punctuation_beginings(&mut tokens);
        let expected: Vec<Token> = vec![
            Token::from("("),
            Token::from("a)"),
            Token::from("\""),
            Token::from("man"),
            Token::from("("),
        ];

        assert_eq!(
            expected,
            tokens,
            "Should split left parens and quotes"
        );
    }
}