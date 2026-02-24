"""Nomad credentials block for authenticating with HashiCorp Nomad clusters."""

from __future__ import annotations

from typing import Any

import nomad
from pydantic import Field, SecretStr

from prefect.blocks.core import Block


class NomadCredentials(Block):
    """Block used to manage authentication with a HashiCorp Nomad cluster.

    Attributes:
        address: The address of the Nomad API (e.g. `http://127.0.0.1:4646`).
        token: The ACL token for authenticating with Nomad.
        namespace: The default Nomad namespace to use.
        region: The default Nomad region to use.
        tls_ca_cert: Path to the CA certificate for TLS verification.
        tls_client_cert: Path to the client certificate for mutual TLS.
        tls_client_key: Path to the client key for mutual TLS.
        tls_skip_verify: Whether to skip TLS certificate verification.
        timeout: Request timeout in seconds.

    Example:
        Load stored Nomad credentials:
        ```python
        from prefect_nomad.credentials import NomadCredentials

        nomad_creds = NomadCredentials.load("my-nomad-creds")
        client = nomad_creds.get_client()
        ```
    """

    _block_type_name = "Nomad Credentials"
    _logo_url = "https://www.datocms-assets.com/2885/1620155117-brandhcnomadprimaryattributedcolor.svg"  # noqa: E501

    address: str = Field(
        default="http://127.0.0.1:4646",
        description="The address of the Nomad API.",
        examples=["http://127.0.0.1:4646", "https://nomad.example.com:4646"],
    )
    token: SecretStr | None = Field(
        default=None,
        description="The ACL token for authenticating with Nomad.",
    )
    namespace: str | None = Field(
        default=None,
        description="The default Nomad namespace to use.",
    )
    region: str | None = Field(
        default=None,
        description="The default Nomad region to use.",
    )
    tls_ca_cert: str | None = Field(
        default=None,
        description="Path to the CA certificate for TLS verification.",
    )
    tls_client_cert: str | None = Field(
        default=None,
        description="Path to the client certificate for mutual TLS.",
    )
    tls_client_key: str | None = Field(
        default=None,
        description="Path to the client key for mutual TLS.",
    )
    tls_skip_verify: bool = Field(
        default=False,
        description="Whether to skip TLS certificate verification.",
    )
    timeout: int = Field(
        default=5,
        description="Request timeout in seconds.",
    )

    def get_client(self) -> nomad.Nomad:
        """Creates and returns a configured `python-nomad` client.

        Returns:
            A `nomad.Nomad` client instance configured with the stored
            credentials and connection parameters.
        """
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
        }

        # Parse host and port from address
        address = self.address
        secure = address.startswith("https://")

        # Strip scheme for python-nomad which adds it based on the `secure` flag
        host = address.replace("https://", "").replace("http://", "")

        # Split host and port
        if ":" in host:
            host_part, port_str = host.rsplit(":", 1)
            try:
                port = int(port_str)
                kwargs["host"] = host_part
                kwargs["port"] = port
            except ValueError:
                kwargs["host"] = host
        else:
            kwargs["host"] = host

        kwargs["secure"] = secure

        if self.token:
            kwargs["token"] = self.token.get_secret_value()

        if self.namespace:
            kwargs["namespace"] = self.namespace

        if self.region:
            kwargs["region"] = self.region

        # TLS configuration
        if secure:
            if self.tls_skip_verify:
                kwargs["verify"] = False
            elif self.tls_ca_cert:
                kwargs["verify"] = self.tls_ca_cert
            else:
                kwargs["verify"] = True

            if self.tls_client_cert and self.tls_client_key:
                kwargs["cert"] = (self.tls_client_cert, self.tls_client_key)
            elif self.tls_client_cert:
                kwargs["cert"] = self.tls_client_cert

        return nomad.Nomad(**kwargs)
