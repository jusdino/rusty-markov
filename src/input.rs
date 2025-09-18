//! input module
//! 
//! Houses logic related to reading in the training corpus
use std::io::BufRead;

/// Reads a BufRead type, line by line
/// 
/// # Example
/// ```rust
/// use std::io::Cursor;
/// use rusty_markov::input::read_lines;
/// 
/// let res = read_lines(Cursor::new("line1\nline2\nline3\n"));
/// assert_eq!(3, res);
/// ```
pub fn read_lines<R: BufRead>(input: R) -> u32 {
    let mut line_count: u32 = 0;

    for line in input.lines() {
        match line {
            Ok(_) => {
                line_count += 1;
            },
            Err(e) => {
                eprintln!("Error reading input: {}", e);
            }
        }
    }

    line_count
}