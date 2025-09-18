//! tokenize module
//! 
//! Contains logic for tokenizing strings


/// Takes an input line of text, returns the line broken up
/// as a vector of tokens
/// 
/// # Example
/// ```rust
/// use rusty_markov::tokenize::tokenize;
/// let input = "I see a little silhouetto of a man.";
/// let tokenized = tokenize(input);
/// 
/// assert_eq!(
///     vec!["I", "see", "a", "little", "silhouetto", "of", "a", "man."],
///     tokenized,
/// )
/// ```
pub fn tokenize(line: &str) -> Vec<String> {
    // Start with just splitting on whitespace
    line.split_whitespace().map(|s| String::from(s)).collect::<Vec<String>>()
}