//! Dimensional analysis macro library
//!
//! Provides compile-time unit checking using the type system.
//! Based on: "Types for Units-of-Measure" (Kennedy, 1994)

use crate::common::Span;
use crate::lexer::TokenKind;
use crate::macro_system::proc_macro::*;
use crate::macro_system::token_tree::*;

/// Dimension representation (SI base units)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Dimension {
    pub length: i8,
    pub mass: i8,
    pub time: i8,
    pub current: i8,
    pub temperature: i8,
    pub amount: i8,
    pub luminosity: i8,
}

impl Dimension {
    pub fn dimensionless() -> Self {
        Self::default()
    }

    pub fn length() -> Self {
        Dimension {
            length: 1,
            ..Default::default()
        }
    }

    pub fn mass() -> Self {
        Dimension {
            mass: 1,
            ..Default::default()
        }
    }

    pub fn time() -> Self {
        Dimension {
            time: 1,
            ..Default::default()
        }
    }

    pub fn current() -> Self {
        Dimension {
            current: 1,
            ..Default::default()
        }
    }

    pub fn temperature() -> Self {
        Dimension {
            temperature: 1,
            ..Default::default()
        }
    }

    pub fn amount() -> Self {
        Dimension {
            amount: 1,
            ..Default::default()
        }
    }

    pub fn luminosity() -> Self {
        Dimension {
            luminosity: 1,
            ..Default::default()
        }
    }

    pub fn mul(&self, other: &Dimension) -> Dimension {
        Dimension {
            length: self.length + other.length,
            mass: self.mass + other.mass,
            time: self.time + other.time,
            current: self.current + other.current,
            temperature: self.temperature + other.temperature,
            amount: self.amount + other.amount,
            luminosity: self.luminosity + other.luminosity,
        }
    }

    pub fn div(&self, other: &Dimension) -> Dimension {
        Dimension {
            length: self.length - other.length,
            mass: self.mass - other.mass,
            time: self.time - other.time,
            current: self.current - other.current,
            temperature: self.temperature - other.temperature,
            amount: self.amount - other.amount,
            luminosity: self.luminosity - other.luminosity,
        }
    }

    pub fn pow(&self, n: i8) -> Dimension {
        Dimension {
            length: self.length * n,
            mass: self.mass * n,
            time: self.time * n,
            current: self.current * n,
            temperature: self.temperature * n,
            amount: self.amount * n,
            luminosity: self.luminosity * n,
        }
    }

    pub fn to_type_tokens(&self) -> TokenStream {
        let mut tokens = TokenStream::new();

        fn int_to_type(n: i8) -> String {
            if n >= 0 {
                format!("P{}", n)
            } else {
                format!("N{}", -n)
            }
        }

        let type_str = format!(
            "Unit<{}, {}, {}, {}, {}, {}, {}>",
            int_to_type(self.length),
            int_to_type(self.mass),
            int_to_type(self.time),
            int_to_type(self.current),
            int_to_type(self.temperature),
            int_to_type(self.amount),
            int_to_type(self.luminosity),
        );

        let token = crate::lexer::Token {
            kind: TokenKind::Ident,
            text: type_str,
            span: Span::default(),
        };
        tokens.push(TokenTree::Token(TokenWithCtx::new(token)));

        tokens
    }
}

/// Common derived units
pub mod derived {
    use super::Dimension;

    pub fn velocity() -> Dimension {
        Dimension::length().div(&Dimension::time())
    }

    pub fn acceleration() -> Dimension {
        velocity().div(&Dimension::time())
    }

    pub fn force() -> Dimension {
        Dimension::mass().mul(&acceleration())
    }

    pub fn energy() -> Dimension {
        force().mul(&Dimension::length())
    }

    pub fn power() -> Dimension {
        energy().div(&Dimension::time())
    }

    pub fn pressure() -> Dimension {
        force().div(&Dimension::length().pow(2))
    }

    pub fn charge() -> Dimension {
        Dimension::current().mul(&Dimension::time())
    }

    pub fn voltage() -> Dimension {
        power().div(&Dimension::current())
    }

    pub fn resistance() -> Dimension {
        voltage().div(&Dimension::current())
    }

    pub fn capacitance() -> Dimension {
        charge().div(&voltage())
    }

    pub fn concentration() -> Dimension {
        Dimension::amount().div(&Dimension::length().pow(3))
    }

    pub fn frequency() -> Dimension {
        Dimension::dimensionless().div(&Dimension::time())
    }
}

/// Pharmacological units
pub mod pharma {
    use super::Dimension;

    pub fn drug_concentration() -> Dimension {
        super::derived::concentration()
    }

    pub fn clearance() -> Dimension {
        Dimension::length().pow(3).div(&Dimension::time())
    }

    pub fn volume_of_distribution() -> Dimension {
        Dimension::length().pow(3).div(&Dimension::mass())
    }

    pub fn half_life() -> Dimension {
        Dimension::time()
    }

    pub fn bioavailability() -> Dimension {
        Dimension::dimensionless()
    }

    pub fn auc() -> Dimension {
        drug_concentration().mul(&Dimension::time())
    }
}

/// Parse a unit name to its dimension
pub fn parse_unit(name: &str) -> Option<Dimension> {
    Some(match name {
        // SI base units
        "m" | "meter" | "meters" => Dimension::length(),
        "kg" | "kilogram" | "kilograms" => Dimension::mass(),
        "s" | "second" | "seconds" => Dimension::time(),
        "A" | "ampere" | "amperes" => Dimension::current(),
        "K" | "kelvin" => Dimension::temperature(),
        "mol" | "mole" | "moles" => Dimension::amount(),
        "cd" | "candela" => Dimension::luminosity(),

        // SI derived units
        "Hz" | "hertz" => derived::frequency(),
        "N" | "newton" | "newtons" => derived::force(),
        "Pa" | "pascal" | "pascals" => derived::pressure(),
        "J" | "joule" | "joules" => derived::energy(),
        "W" | "watt" | "watts" => derived::power(),
        "C" | "coulomb" | "coulombs" => derived::charge(),
        "V" | "volt" | "volts" => derived::voltage(),
        "Ω" | "ohm" | "ohms" => derived::resistance(),
        "F" | "farad" | "farads" => derived::capacitance(),

        // Common prefixed units
        "mm" | "millimeter" => Dimension::length(),
        "cm" | "centimeter" => Dimension::length(),
        "km" | "kilometer" => Dimension::length(),
        "g" | "gram" => Dimension::mass(),
        "mg" | "milligram" => Dimension::mass(),
        "μg" | "microgram" => Dimension::mass(),
        "ms" | "millisecond" => Dimension::time(),
        "μs" | "microsecond" => Dimension::time(),
        "ns" | "nanosecond" => Dimension::time(),
        "min" | "minute" => Dimension::time(),
        "h" | "hour" => Dimension::time(),

        // Pharmacological units
        "mL" | "milliliter" => Dimension::length().pow(3),
        "L" | "liter" => Dimension::length().pow(3),
        "mmol" => Dimension::amount(),
        "μmol" => Dimension::amount(),
        "mM" | "millimolar" => pharma::drug_concentration(),
        "μM" | "micromolar" => pharma::drug_concentration(),
        "nM" | "nanomolar" => pharma::drug_concentration(),

        // Dimensionless
        "percent" | "%" => Dimension::dimensionless(),
        "ppm" => Dimension::dimensionless(),

        _ => return None,
    })
}

/// Unit macro: unit!(value: unit_name)
pub fn expand_unit_macro(input: TokenStream) -> Result<TokenStream, ProcMacroError> {
    let trees = input.into_trees();

    if trees.len() < 3 {
        return Err(ProcMacroError::new("expected: value : unit"));
    }

    let value = &trees[0];

    let unit_name = match trees.last() {
        Some(TokenTree::Token(t)) if t.token.kind == TokenKind::Ident => &t.token.text,
        _ => return Err(ProcMacroError::new("expected unit name")),
    };

    let dimension = parse_unit(unit_name)
        .ok_or_else(|| ProcMacroError::new(format!("unknown unit: {}", unit_name)))?;

    let mut result = TokenStream::new();

    result.push(make_ident("Quantity"));
    result.push(make_punct("::"));
    result.push(make_punct("<"));
    result.extend(dimension.to_type_tokens());
    result.push(make_punct(">"));
    result.push(make_punct("::"));
    result.push(make_ident("new"));
    result.push(make_punct("("));
    result.push(value.clone());
    result.push(make_punct(")"));

    Ok(result)
}

fn make_ident(name: &str) -> TokenTree {
    let token = crate::lexer::Token {
        kind: TokenKind::Ident,
        text: name.to_string(),
        span: crate::common::Span::default(),
    };
    TokenTree::Token(TokenWithCtx::new(token))
}

fn make_punct(p: &str) -> TokenTree {
    let kind = match p {
        "::" => TokenKind::ColonColon,
        "<" => TokenKind::Lt,
        ">" => TokenKind::Gt,
        "(" => TokenKind::LParen,
        ")" => TokenKind::RParen,
        "+" => TokenKind::Plus,
        "-" => TokenKind::Minus,
        "*" => TokenKind::Star,
        "/" => TokenKind::Slash,
        "," => TokenKind::Comma,
        ";" => TokenKind::Semi,
        ":" => TokenKind::Colon,
        "." => TokenKind::Dot,
        _ => TokenKind::Ident, // Fallback for unknown punctuation
    };
    let token = crate::lexer::Token {
        kind,
        text: p.to_string(),
        span: crate::common::Span::default(),
    };
    TokenTree::Token(TokenWithCtx::new(token))
}
