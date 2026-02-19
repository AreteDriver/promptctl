"""Exception hierarchy for promptctl."""


class PromptctlError(Exception):
    """Base exception for all promptctl errors."""


class ConfigError(PromptctlError):
    """Configuration read/write/validation failed."""


class ClientError(PromptctlError):
    """Anthropic API communication failed."""


class PromptError(PromptctlError):
    """Prompt template loading or execution failed."""


class ReviewError(PromptctlError):
    """Code review operation failed."""


class LicenseError(PromptctlError):
    """License validation failed."""
