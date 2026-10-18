use core::fmt;

/// Codec failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Cooperative cancellation or deadline expiry.
    #[cfg(feature = "encoder")]
    Cancelled(enough::StopReason),
    /// Invalid encoder settings or buffer geometry.
    InvalidArgument(&'static str),
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
            #[cfg(feature = "encoder")]
            Error::Cancelled(reason) => write!(f, "operation stopped: {reason}"),
            Error::InvalidArgument(what) => write!(f, "invalid argument: {what}"),
            Error::Truncated(what) => write!(f, "truncated bitstream: {what}"),
            Error::Invalid(what) => write!(f, "invalid bitstream: {what}"),
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
            Error::LimitExceeded(what) => write!(f, "limit exceeded: {what}"),
        }
    }
}

impl core::error::Error for Error {}

#[cfg(feature = "encoder")]
impl From<enough::StopReason> for Error {
    fn from(reason: enough::StopReason) -> Self {
        Self::Cancelled(reason)
    }
}

pub type Result<T> = core::result::Result<T, Error>;
