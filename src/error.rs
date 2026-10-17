use core::fmt;

/// Decoding failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The bitstream ended before a field or payload was complete.
    Truncated(&'static str),
    /// A field holds a value the format does not allow.
    Invalid(&'static str),
    /// A valid stream uses something this port does not implement yet.
    Unsupported(&'static str),
    /// The stream asks for more memory or pixels than the limits allow.
    LimitExceeded(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated(what) => write!(f, "truncated bitstream: {what}"),
            Error::Invalid(what) => write!(f, "invalid bitstream: {what}"),
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
            Error::LimitExceeded(what) => write!(f, "limit exceeded: {what}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
