use std::collections::VecDeque;

use crate::token::Token;


/// Trait for converting a token into a context vector
pub trait AsContext {
    /// Convert the token into a context vector
    fn as_context(&self) -> Vec<Token>;
}


impl AsContext for &Token {
    fn as_context(&self) -> Vec<Token> {
        vec![(*self).clone()]
    }
}

impl AsContext for &Vec<Token> {
    fn as_context(&self) -> Vec<Token> {
        self.to_vec()
    }
}

impl AsContext for &[Token] {
    fn as_context(&self) -> Vec<Token> {
        self.to_vec()
    }
}

impl AsContext for &VecDeque<Token> {
    fn as_context(&self) -> Vec<Token> {
        self.iter().cloned().collect()
    }
}


#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use crate::token::Token;
    use crate::as_context::AsContext;

    #[test]
    fn test_vecdeque_push_front() {
        let mut v = VecDeque::new();
        v.push_front(Token::from("one"));
        v.push_front(Token::from("two"));
        v.push_front(Token::from("three"));

        // push_front adds to the front, so order is [three, two, one]
        assert_eq!(
            vec![
                Token::from("three"),
                Token::from("two"),
                Token::from("one"),
            ],
            (&v).as_context()
        )
    }

    #[test]
    fn test_vecdeque_push_back() {
        let mut v = VecDeque::new();
        v.push_back(Token::from("one"));
        v.push_back(Token::from("two"));
        v.push_back(Token::from("three"));

        // push_back adds to the back, so order is [one, two, three]
        assert_eq!(
            vec![
                Token::from("one"),
                Token::from("two"),
                Token::from("three"),
            ],
            (&v).as_context()
        )
    }

    #[test]
    fn test_token() {
        let token = Token::from("hello");
        
        // A single token should convert to a vec with that token
        assert_eq!(
            vec![Token::from("hello")],
            (&token).as_context()
        )
    }

    #[test]
    fn test_token_boundary() {
        let token = Token::Boundary(String::from("."));
        
        // A boundary token should also convert to a vec with that token
        assert_eq!(
            vec![Token::Boundary(String::from("."))],
            (&token).as_context()
        )
    }

    #[test]
    fn test_vec_token() {
        let vec = vec![
            Token::from("one"),
            Token::from("two"),
            Token::from("three"),
        ];
        
        // Vec should convert to a new vec with the same tokens
        assert_eq!(
            vec![
                Token::from("one"),
                Token::from("two"),
                Token::from("three"),
            ],
            (&vec).as_context()
        )
    }

    #[test]
    fn test_slice_token_partial() {
        let vec = vec![
            Token::from("one"),
            Token::from("two"),
            Token::from("three"),
        ];
        let slice = &vec[1..3]; // Just "two" and "three"
        
        // Partial slice should convert to a vec with just those tokens
        assert_eq!(
            vec![
                Token::from("two"),
                Token::from("three"),
            ],
            slice.as_context()
        )
    }
}
