#[derive(Eq, Hash, Debug, Clone)]
pub enum Token {
    Token(String),
    Boundary(String),
}

#[cfg(feature = "memory-profiling")]
use memuse::DynamicUsage;

#[cfg(feature = "memory-profiling")]
impl DynamicUsage for Token {
    fn dynamic_usage(&self) -> usize {
        match self {
            Token::Token(s) => s.capacity(),
            Token::Boundary => std::mem::size_of::<Token>(),
        }
    }
    
    fn dynamic_usage_bounds(&self) -> (usize, Option<usize>) {
        let usage = self.dynamic_usage();
        (usage, Some(usage))
    }
}

impl Token {
    pub fn from<S: Into<String>>(value: S) -> Token {
        Token::Token(value.into())
    }
}

impl PartialEq for Token {
    fn eq(&self, other: &Token) -> bool {
        match (self, other) {
            (Token::Token(s), Token::Token(o)) => s == o,
            // Consider all boundaries equal
            (Token::Boundary(_), Token::Boundary(_)) => true,
            _ => false,
        }
    }
}
