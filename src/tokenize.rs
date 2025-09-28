//! tokenize module
//!
//! Contains logic for tokenizing strings

use crate::token::Token;


/// Takes an input line of text, returns the line broken up
/// as a vector of tokens
pub fn tokenize(line: &str) -> impl Iterator<Item = Token> {
    // Start with just splitting on whitespace
    line.split_whitespace().map(|s| Token::from(s))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {

        let input = "I see a little silhouetto of a man.";
        let tokenized = tokenize(input);
        let output: Vec<Token> = vec!["I", "see", "a", "little", "silhouetto", "of", "a", "man."].iter().map(|s| Token::from(*s)).collect();

        assert_eq!(
            output,
            tokenized.collect::<Vec<Token>>(),
        )
    }
}