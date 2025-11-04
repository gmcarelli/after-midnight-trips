from typing import Protocol, TypeVar, Generic

TClient = TypeVar("TClient", covariant=True)


class HostConnector(Protocol, Generic[TClient]):
    def connect_to_host(self, host_url: str) -> TClient: ...
