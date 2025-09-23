//! tokenize module
//!
//! Contains logic for tokenizing strings


/// Takes an input line of text, returns the line broken up
/// as a vector of tokens
pub fn tokenize(line: &str) -> impl Iterator<Item = String> {
    // Start with just splitting on whitespace
    line.split_whitespace().map(|s| String::from(s))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {

        let input = "I see a little silhouetto of a man.";
        let tokenized = tokenize(input);

        assert_eq!(
            vec!["I", "see", "a", "little", "silhouetto", "of", "a", "man."],
            tokenized.collect::<Vec<String>>(),
        )
    }
}