"""Tests for the NomadCredentials block."""

from unittest.mock import MagicMock, patch

from prefect_nomad.credentials import NomadCredentials


class TestNomadCredentialsFields:
    """Test default field values and validation."""

    def test_default_address(self):
        creds = NomadCredentials()
        assert creds.address == "http://127.0.0.1:4646"

    def test_default_token_is_none(self):
        creds = NomadCredentials()
        assert creds.token is None

    def test_default_namespace_is_none(self):
        creds = NomadCredentials()
        assert creds.namespace is None

    def test_default_region_is_none(self):
        creds = NomadCredentials()
        assert creds.region is None

    def test_default_tls_skip_verify_is_false(self):
        creds = NomadCredentials()
        assert creds.tls_skip_verify is False

    def test_default_timeout(self):
        creds = NomadCredentials()
        assert creds.timeout == 5

    def test_custom_address(self):
        creds = NomadCredentials(address="https://nomad.example.com:4646")
        assert creds.address == "https://nomad.example.com:4646"

    def test_secret_token_is_masked(self):
        creds = NomadCredentials(token="super-secret-token")
        assert "super-secret-token" not in str(creds.token)
        assert creds.token.get_secret_value() == "super-secret-token"


class TestNomadCredentialsGetClient:
    """Test the get_client() method."""

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_default_get_client(self, mock_nomad_cls):
        """get_client() with defaults passes host, port, secure, and timeout."""
        creds = NomadCredentials()
        creds.get_client()

        mock_nomad_cls.assert_called_once()
        kwargs = mock_nomad_cls.call_args.kwargs
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 4646
        assert kwargs["secure"] is False
        assert kwargs["timeout"] == 5

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_with_token(self, mock_nomad_cls):
        creds = NomadCredentials(token="my-token")
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert kwargs["token"] == "my-token"

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_without_token(self, mock_nomad_cls):
        creds = NomadCredentials()
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert "token" not in kwargs

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_with_namespace(self, mock_nomad_cls):
        creds = NomadCredentials(namespace="production")
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert kwargs["namespace"] == "production"

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_with_region(self, mock_nomad_cls):
        creds = NomadCredentials(region="us-east-1")
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert kwargs["region"] == "us-east-1"

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_https_with_verify(self, mock_nomad_cls):
        creds = NomadCredentials(address="https://nomad.example.com:4646")
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert kwargs["secure"] is True
        assert kwargs["verify"] is True

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_https_skip_verify(self, mock_nomad_cls):
        creds = NomadCredentials(
            address="https://nomad.example.com:4646",
            tls_skip_verify=True,
        )
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert kwargs["verify"] is False

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_https_with_ca_cert(self, mock_nomad_cls):
        creds = NomadCredentials(
            address="https://nomad.example.com:4646",
            tls_ca_cert="/path/to/ca.pem",
        )
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert kwargs["verify"] == "/path/to/ca.pem"

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_https_with_client_certs(self, mock_nomad_cls):
        creds = NomadCredentials(
            address="https://nomad.example.com:4646",
            tls_client_cert="/path/to/cert.pem",
            tls_client_key="/path/to/key.pem",
        )
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert kwargs["cert"] == ("/path/to/cert.pem", "/path/to/key.pem")

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_https_with_client_cert_only(self, mock_nomad_cls):
        creds = NomadCredentials(
            address="https://nomad.example.com:4646",
            tls_client_cert="/path/to/cert.pem",
        )
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert kwargs["cert"] == "/path/to/cert.pem"

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_http_no_tls_kwargs(self, mock_nomad_cls):
        """HTTP addresses should not set verify or cert kwargs."""
        creds = NomadCredentials(
            address="http://127.0.0.1:4646",
            tls_ca_cert="/path/to/ca.pem",  # Should be ignored for HTTP
        )
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert "verify" not in kwargs
        assert "cert" not in kwargs

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_address_without_port(self, mock_nomad_cls):
        """Address without port should set host only."""
        creds = NomadCredentials(address="http://nomad.local")
        creds.get_client()

        kwargs = mock_nomad_cls.call_args.kwargs
        assert kwargs["host"] == "nomad.local"
        assert "port" not in kwargs

    @patch("prefect_nomad.credentials.nomad.Nomad")
    def test_get_client_returns_nomad_instance(self, mock_nomad_cls):
        mock_instance = MagicMock()
        mock_nomad_cls.return_value = mock_instance

        creds = NomadCredentials()
        result = creds.get_client()

        assert result is mock_instance
