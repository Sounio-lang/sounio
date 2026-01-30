//! Session types for epistemic protocols
//!
//! Session types encode communication protocols at the type level,
//! ensuring that protocol participants follow the correct sequence
//! of operations. Based on Gay & Hole (1999) and Honda et al.
//!
//! In the epistemic context, session types model:
//! - Query-response protocols for knowledge bases
//! - Bayesian update protocols
//! - Peer review processes
//! - Clinical trial data collection
//!
//! Key features:
//! - Duality: Each session type has a dual (the other party's view)
//! - Linearity: Sessions must be used exactly once (no protocol abandon)
//! - Progress: Well-typed sessions guarantee progress (no deadlock)

use std::fmt;
use std::sync::Arc;

use super::linear_types::LinearType;

/// Session type for epistemic protocols
#[derive(Clone, Debug)]
pub enum SessionType {
    /// Send a value and continue
    Send {
        payload: Arc<LinearType>,
        continuation: Arc<SessionType>,
    },

    /// Receive a value and continue
    Recv {
        payload: Arc<LinearType>,
        continuation: Arc<SessionType>,
    },

    /// Offer a choice (server-side / internal choice)
    ///
    /// The session offers multiple branches; the peer chooses.
    Offer {
        branches: Vec<(String, Arc<SessionType>)>,
    },

    /// Make a choice (client-side / external choice)
    ///
    /// This session chooses which branch to take.
    Choose {
        branches: Vec<(String, Arc<SessionType>)>,
    },

    /// End of session
    End,

    /// Recursive session type
    Rec { var: String, body: Arc<SessionType> },

    /// Recursion variable
    Var(String),
}

impl SessionType {
    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create a send action
    pub fn send(payload: LinearType, continuation: SessionType) -> Self {
        SessionType::Send {
            payload: Arc::new(payload),
            continuation: Arc::new(continuation),
        }
    }

    /// Create a receive action
    pub fn recv(payload: LinearType, continuation: SessionType) -> Self {
        SessionType::Recv {
            payload: Arc::new(payload),
            continuation: Arc::new(continuation),
        }
    }

    /// Create an offer (internal choice)
    pub fn offer(branches: Vec<(&str, SessionType)>) -> Self {
        SessionType::Offer {
            branches: branches
                .into_iter()
                .map(|(l, s)| (l.to_string(), Arc::new(s)))
                .collect(),
        }
    }

    /// Create a choice (external choice)
    pub fn choose(branches: Vec<(&str, SessionType)>) -> Self {
        SessionType::Choose {
            branches: branches
                .into_iter()
                .map(|(l, s)| (l.to_string(), Arc::new(s)))
                .collect(),
        }
    }

    /// Create a binary offer
    pub fn offer_binary(left: SessionType, right: SessionType) -> Self {
        Self::offer(vec![("left", left), ("right", right)])
    }

    /// Create a binary choice
    pub fn choose_binary(left: SessionType, right: SessionType) -> Self {
        Self::choose(vec![("left", left), ("right", right)])
    }

    /// Create a recursive session
    pub fn rec(var: &str, body: SessionType) -> Self {
        SessionType::Rec {
            var: var.to_string(),
            body: Arc::new(body),
        }
    }

    /// Create a recursion variable
    pub fn var(name: &str) -> Self {
        SessionType::Var(name.to_string())
    }

    // ========================================================================
    // Duality
    // ========================================================================

    /// Compute the dual of this session type
    ///
    /// The dual represents the other party's view of the protocol.
    /// ```text
    /// dual(Send A.S) = Recv A.dual(S)
    /// dual(Recv A.S) = Send A.dual(S)
    /// dual(Offer{l:S}) = Choose{l:dual(S)}
    /// dual(Choose{l:S}) = Offer{l:dual(S)}
    /// dual(End) = End
    /// dual(μX.S) = μX.dual(S)
    /// dual(X) = X
    /// ```
    pub fn dual(&self) -> SessionType {
        match self {
            SessionType::Send {
                payload,
                continuation,
            } => SessionType::Recv {
                payload: payload.clone(),
                continuation: Arc::new(continuation.dual()),
            },
            SessionType::Recv {
                payload,
                continuation,
            } => SessionType::Send {
                payload: payload.clone(),
                continuation: Arc::new(continuation.dual()),
            },
            SessionType::Offer { branches } => SessionType::Choose {
                branches: branches
                    .iter()
                    .map(|(l, s)| (l.clone(), Arc::new(s.dual())))
                    .collect(),
            },
            SessionType::Choose { branches } => SessionType::Offer {
                branches: branches
                    .iter()
                    .map(|(l, s)| (l.clone(), Arc::new(s.dual())))
                    .collect(),
            },
            SessionType::End => SessionType::End,
            SessionType::Rec { var, body } => SessionType::Rec {
                var: var.clone(),
                body: Arc::new(body.dual()),
            },
            SessionType::Var(v) => SessionType::Var(v.clone()),
        }
    }

    /// Check if this session is the dual of another
    pub fn is_dual_of(&self, other: &SessionType) -> bool {
        self.definitionally_equal(&other.dual())
    }

    // ========================================================================
    // Structural operations
    // ========================================================================

    /// Check if this is the End session
    pub fn is_end(&self) -> bool {
        matches!(self, SessionType::End)
    }

    /// Check structural equality
    pub fn definitionally_equal(&self, other: &SessionType) -> bool {
        match (self, other) {
            (
                SessionType::Send {
                    payload: p1,
                    continuation: c1,
                },
                SessionType::Send {
                    payload: p2,
                    continuation: c2,
                },
            ) => p1.definitionally_equal(p2) && c1.definitionally_equal(c2),

            (
                SessionType::Recv {
                    payload: p1,
                    continuation: c1,
                },
                SessionType::Recv {
                    payload: p2,
                    continuation: c2,
                },
            ) => p1.definitionally_equal(p2) && c1.definitionally_equal(c2),

            (SessionType::Offer { branches: b1 }, SessionType::Offer { branches: b2 }) => {
                if b1.len() != b2.len() {
                    return false;
                }
                b1.iter()
                    .zip(b2.iter())
                    .all(|((l1, s1), (l2, s2))| l1 == l2 && s1.definitionally_equal(s2))
            }

            (SessionType::Choose { branches: b1 }, SessionType::Choose { branches: b2 }) => {
                if b1.len() != b2.len() {
                    return false;
                }
                b1.iter()
                    .zip(b2.iter())
                    .all(|((l1, s1), (l2, s2))| l1 == l2 && s1.definitionally_equal(s2))
            }

            (SessionType::End, SessionType::End) => true,

            (SessionType::Rec { var: v1, body: b1 }, SessionType::Rec { var: v2, body: b2 }) => {
                v1 == v2 && b1.definitionally_equal(b2)
            }

            (SessionType::Var(v1), SessionType::Var(v2)) => v1 == v2,

            _ => false,
        }
    }

    /// Unfold a recursive session type
    pub fn unfold(&self) -> SessionType {
        match self {
            SessionType::Rec { var, body } => body.substitute(var, self),
            _ => self.clone(),
        }
    }

    /// Substitute a variable in the session type
    pub fn substitute(&self, var: &str, replacement: &SessionType) -> SessionType {
        match self {
            SessionType::Var(v) if v == var => replacement.clone(),
            SessionType::Send {
                payload,
                continuation,
            } => SessionType::Send {
                payload: payload.clone(),
                continuation: Arc::new(continuation.substitute(var, replacement)),
            },
            SessionType::Recv {
                payload,
                continuation,
            } => SessionType::Recv {
                payload: payload.clone(),
                continuation: Arc::new(continuation.substitute(var, replacement)),
            },
            SessionType::Offer { branches } => SessionType::Offer {
                branches: branches
                    .iter()
                    .map(|(l, s)| (l.clone(), Arc::new(s.substitute(var, replacement))))
                    .collect(),
            },
            SessionType::Choose { branches } => SessionType::Choose {
                branches: branches
                    .iter()
                    .map(|(l, s)| (l.clone(), Arc::new(s.substitute(var, replacement))))
                    .collect(),
            },
            SessionType::Rec { var: v, body } if v != var => SessionType::Rec {
                var: v.clone(),
                body: Arc::new(body.substitute(var, replacement)),
            },
            _ => self.clone(),
        }
    }

    // ========================================================================
    // Common Protocols
    // ========================================================================

    /// Query-response protocol
    ///
    /// Send query, receive response, end.
    pub fn query_response(query: LinearType, response: LinearType) -> Self {
        SessionType::send(query, SessionType::recv(response, SessionType::End))
    }

    /// Bayesian update protocol
    ///
    /// Send evidence, receive updated belief, choose to confirm or reject.
    pub fn bayesian_update(evidence: LinearType, belief: LinearType, reason: LinearType) -> Self {
        SessionType::send(
            evidence,
            SessionType::recv(
                belief,
                SessionType::choose_binary(
                    SessionType::End,                            // Confirm
                    SessionType::send(reason, SessionType::End), // Reject with reason
                ),
            ),
        )
    }

    /// Streaming protocol (potentially infinite)
    ///
    /// Either send more data and continue, or end.
    pub fn stream(data: LinearType) -> Self {
        SessionType::rec(
            "Stream",
            SessionType::offer_binary(
                SessionType::send(data, SessionType::var("Stream")),
                SessionType::End,
            ),
        )
    }

    /// Peer review protocol
    ///
    /// Receive manuscript, then loop: either request revision, accept, or reject.
    pub fn peer_review(
        manuscript: LinearType,
        revision: LinearType,
        accept: LinearType,
        reject: LinearType,
    ) -> Self {
        SessionType::recv(
            manuscript.clone(),
            SessionType::rec(
                "Review",
                SessionType::offer(vec![
                    (
                        "revise",
                        SessionType::send(
                            revision.clone(),
                            SessionType::recv(manuscript, SessionType::var("Review")),
                        ),
                    ),
                    ("accept", SessionType::send(accept, SessionType::End)),
                    ("reject", SessionType::send(reject, SessionType::End)),
                ]),
            ),
        )
    }
}

impl fmt::Display for SessionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionType::Send {
                payload,
                continuation,
            } => {
                write!(f, "!{}.{}", payload, continuation)
            }
            SessionType::Recv {
                payload,
                continuation,
            } => {
                write!(f, "?{}.{}", payload, continuation)
            }
            SessionType::Offer { branches } => {
                write!(f, "&{{")?;
                for (i, (label, session)) in branches.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", label, session)?;
                }
                write!(f, "}}")
            }
            SessionType::Choose { branches } => {
                write!(f, "⊕{{")?;
                for (i, (label, session)) in branches.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", label, session)?;
                }
                write!(f, "}}")
            }
            SessionType::End => write!(f, "end"),
            SessionType::Rec { var, body } => write!(f, "μ{}.{}", var, body),
            SessionType::Var(v) => write!(f, "{}", v),
        }
    }
}

/// Protocol error
#[derive(Clone, Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("Unexpected send: session expected receive or end")]
    UnexpectedSend,

    #[error("Unexpected receive: session expected send or end")]
    UnexpectedRecv,

    #[error("Unexpected choice: session doesn't offer choices")]
    UnexpectedChoice,

    #[error("Invalid choice label: {0}")]
    InvalidChoice(String),

    #[error("Session not complete: expected End")]
    SessionNotComplete,

    #[error("Duality violation: sessions are not dual")]
    DualityViolation,

    #[error("Protocol type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },
}

/// Session type checker / runtime
pub struct SessionChecker {
    /// Current session state
    current: SessionType,
}

impl SessionChecker {
    /// Create a new session checker
    pub fn new(session: SessionType) -> Self {
        Self { current: session }
    }

    /// Get current state
    pub fn current(&self) -> &SessionType {
        &self.current
    }

    /// Process a send operation
    pub fn send(&mut self, value_type: &LinearType) -> Result<(), ProtocolError> {
        // Unfold if recursive
        let unfolded = self.current.unfold();

        match unfolded {
            SessionType::Send {
                payload,
                continuation,
            } => {
                // Check type compatibility (simplified)
                if !payload.definitionally_equal(value_type) {
                    return Err(ProtocolError::TypeMismatch {
                        expected: format!("{}", payload),
                        found: format!("{}", value_type),
                    });
                }
                self.current = (*continuation).clone();
                Ok(())
            }
            _ => Err(ProtocolError::UnexpectedSend),
        }
    }

    /// Process a receive operation
    pub fn recv(&mut self) -> Result<LinearType, ProtocolError> {
        let unfolded = self.current.unfold();

        match unfolded {
            SessionType::Recv {
                payload,
                continuation,
            } => {
                let typ = (*payload).clone();
                self.current = (*continuation).clone();
                Ok(typ)
            }
            _ => Err(ProtocolError::UnexpectedRecv),
        }
    }

    /// Make a choice (for Choose session type)
    pub fn choose(&mut self, label: &str) -> Result<(), ProtocolError> {
        let unfolded = self.current.unfold();

        match unfolded {
            SessionType::Choose { branches } => {
                for (l, s) in branches {
                    if l == label {
                        self.current = (*s).clone();
                        return Ok(());
                    }
                }
                Err(ProtocolError::InvalidChoice(label.to_string()))
            }
            _ => Err(ProtocolError::UnexpectedChoice),
        }
    }

    /// Respond to an offer (for Offer session type from peer's view)
    pub fn offer(&mut self, label: &str) -> Result<(), ProtocolError> {
        let unfolded = self.current.unfold();

        match unfolded {
            SessionType::Offer { branches } => {
                for (l, s) in branches {
                    if l == label {
                        self.current = (*s).clone();
                        return Ok(());
                    }
                }
                Err(ProtocolError::InvalidChoice(label.to_string()))
            }
            _ => Err(ProtocolError::UnexpectedChoice),
        }
    }

    /// Close the session (must be at End)
    pub fn close(self) -> Result<(), ProtocolError> {
        if self.current.is_end() {
            Ok(())
        } else {
            Err(ProtocolError::SessionNotComplete)
        }
    }

    /// Check if session is complete
    pub fn is_complete(&self) -> bool {
        self.current.is_end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_send_recv() {
        let send = SessionType::send(LinearType::One, SessionType::End);
        let dual = send.dual();

        match dual {
            SessionType::Recv { continuation, .. } => {
                assert!(continuation.is_end());
            }
            _ => panic!("Expected Recv"),
        }
    }

    #[test]
    fn test_dual_involution() {
        let session = SessionType::send(
            LinearType::One,
            SessionType::recv(LinearType::Top, SessionType::End),
        );

        let dual1 = session.dual();
        let dual2 = dual1.dual();

        assert!(session.definitionally_equal(&dual2));
    }

    #[test]
    fn test_dual_offer_choose() {
        let offer = SessionType::offer_binary(SessionType::End, SessionType::End);

        match offer.dual() {
            SessionType::Choose { .. } => {}
            _ => panic!("Expected Choose"),
        }
    }

    #[test]
    fn test_query_response_protocol() {
        let query = LinearType::One;
        let response = LinearType::Top;
        let protocol = SessionType::query_response(query.clone(), response.clone());

        // Create checker
        let mut checker = SessionChecker::new(protocol);

        // Send query
        checker.send(&query).unwrap();

        // Receive response
        let received = checker.recv().unwrap();
        assert!(received.definitionally_equal(&response));

        // Should be at end
        checker.close().unwrap();
    }

    #[test]
    fn test_recursive_unfold() {
        let stream = SessionType::stream(LinearType::One);

        // Unfold once
        let unfolded = stream.unfold();

        // Should be Offer with send branch and end branch
        match unfolded {
            SessionType::Offer { branches } => {
                assert_eq!(branches.len(), 2);
            }
            _ => panic!("Expected Offer after unfold"),
        }
    }

    #[test]
    fn test_session_checker_basic() {
        let protocol = SessionType::send(LinearType::One, SessionType::End);
        let mut checker = SessionChecker::new(protocol);

        // Send
        checker.send(&LinearType::One).unwrap();

        // Should be complete
        assert!(checker.is_complete());
    }

    #[test]
    fn test_session_checker_wrong_order() {
        let protocol = SessionType::recv(LinearType::One, SessionType::End);
        let mut checker = SessionChecker::new(protocol);

        // Try to send when we should receive - should fail
        let result = checker.send(&LinearType::One);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_dual() {
        let client = SessionType::send(LinearType::One, SessionType::End);
        let server = SessionType::recv(LinearType::One, SessionType::End);

        assert!(client.is_dual_of(&server));
        assert!(server.is_dual_of(&client));
    }
}
