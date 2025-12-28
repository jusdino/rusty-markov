# rusty-markov
A project to learn Rust by making a [markov-chain](https://en.wikipedia.org/wiki/Markov_chain) text generator.

This project accepts a text training corpus, streamed in via stdin, uses the sequence of words in the corpus to train
a probability model for what 'token' is likely to follow a given sequence of 'context' tokens. It then uses that
probability model to generate some random text that sounds vageuely like the content of the training corpus.

Try it:
```sh
cargo run -- --boundaries sentence-endings --order 2 <corpus-examples/moby-dick.txt 2>/dev/null
The Parsee!” ” cried Ahab , in one corner ; when all at once to the forecastle .
```

For cli help:
```sh
$ cargo run -- --help
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.02s
     Running `target/debug/rusty-markov --help`
A Markov chain text generator

Usage: rusty-markov [OPTIONS]

Options:
  -m, --max-tokens <MAX_TOKENS>
          Number of tokens to generate

          [default: 100]

  -b, --boundaries <BOUNDARIES>
          Boundary configuration for training

          Possible values:
          - line-endings:     Line endings are boundaries (like in a play transcript)
          - sentence-endings: Sentence endings are boundaries (like most anything else)

          [default: line-endings]

  -o, --order <ORDER>
          Chain context order
          
          [default: 3]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

> **Note:** The implementation in this project is likely _not_ to follow academic discussion of algorithms in natural language processing. I'm deliberately not researching the subject as a fun exercise to explore the concept on my own. Sometimes it's fun to take on a project like this, starting from a place of ignorance, and seeing where it takes you.
