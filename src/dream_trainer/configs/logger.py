from dataclasses import dataclass


@dataclass(kw_only=True)
class LoggingParameters:
    enabled: bool = True


class WandbLoggingParameters(LoggingParameters): ...
