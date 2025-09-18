//! babble module
//! 
//! Contains logic for producing text based on a given probability map
use std::collections::HashMap;

use rand::seq::IteratorRandom;
use rand;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

/// Print generated text to stdout
pub fn babble(probability: &HashMap<String, HashMap<String, u32>>, limit: u32) {
    let output = generate_text(probability, limit);
    let transformed: Vec<String> = output.iter().map(|s| String::from(*s)).collect();

    println!("{}", transformed.join(" "))
}


/// Generate some text based on the probability mapping
/// 
/// # Example
/// ```rust
/// use 
/// ```
pub fn generate_text<'a>(probability: &'a HashMap<String, HashMap<String, u32>>, limit: u32) -> Vec<&'a String> {
    let mut output = Vec::new();

    let mut rng = rand::rng();
    let mut last_token = match probability.keys().choose(&mut rng) {
        Some(key) => key,
        None => {
            eprintln!("Empty probability map provided");
            return output;
        }
    };
    output.push(last_token);

    // Generate up to limit tokens, starting with `last_token`
    for _ in 0..limit {
        let next_token_probs = match probability.get(last_token) {
            Some(p) => p,
            None => {
                // If last_token is not in our probabilities, break early to end generation
                break;
            }
        };
        
        let (probs, tokens) = decompose_probabilities(next_token_probs);
        let dist = WeightedIndex::new(probs).unwrap();
        let next_token = tokens[dist.sample(&mut rng)];
        output.push(next_token);

        // Shift tokens for next iteration
        last_token = next_token;
    }

    output
}


fn decompose_probabilities(prob_map: &HashMap<String, u32>) -> (Vec<u32>, Vec<&String>) {
    let mut probs = Vec::new();
    let mut tokens = Vec::new();

    for (k, v) in prob_map.iter() {
        tokens.push(k);
        probs.push(*v);
    }

    (probs, tokens)
}
