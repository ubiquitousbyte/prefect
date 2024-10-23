import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Iterable,
    Mapping,
    Optional,
    Union,
)
from urllib.parse import urlparse

from pydantic import BeforeValidator, Field, SecretStr, model_validator
from pydantic_settings import SettingsConfigDict
from typing_extensions import Self

from prefect.settings.base import PrefectBaseSettings
from prefect.types import LogLevel
from prefect.utilities.collections import deep_merge_dicts, set_in_dict

from .api import APISettings
from .cli import CLISettings
from .client import ClientSettings
from .cloud import CloudSettings
from .deployments import DeploymentsSettings
from .flows import FlowsSettings
from .logging import LoggingSettings
from .results import ResultsSettings
from .runner import RunnerSettings
from .server import ServerSettings

if TYPE_CHECKING:
    from prefect.settings.legacy import Setting


class Settings(PrefectBaseSettings):
    """
    Settings for Prefect using Pydantic settings.

    See https://docs.pydantic.dev/latest/concepts/pydantic_settings
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PREFECT_",
        env_nested_delimiter=None,
        extra="ignore",
    )

    api: APISettings = Field(
        default_factory=APISettings,
        description="Settings for interacting with the Prefect API",
    )

    cli: CLISettings = Field(
        default_factory=CLISettings,
        description="Settings for controlling CLI behavior",
    )

    client: ClientSettings = Field(
        default_factory=ClientSettings,
        description="Settings for for controlling API client behavior",
    )

    cloud: CloudSettings = Field(
        default_factory=CloudSettings,
        description="Settings for interacting with Prefect Cloud",
    )

    deployments: DeploymentsSettings = Field(
        default_factory=DeploymentsSettings,
        description="Settings for configuring deployments defaults",
    )

    flows: FlowsSettings = Field(
        default_factory=FlowsSettings,
        description="Settings for controlling flow behavior",
    )

    logging: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="Settings for controlling logging behavior",
    )

    results: ResultsSettings = Field(
        default_factory=ResultsSettings,
        description="Settings for controlling result storage behavior",
    )

    runner: RunnerSettings = Field(
        default_factory=RunnerSettings,
        description="Settings for controlling runner behavior",
    )

    server: ServerSettings = Field(
        default_factory=ServerSettings,
        description="Settings for controlling server behavior",
    )

    ###########################################################################
    # Testing

    test_mode: bool = Field(
        default=False,
        description="If `True`, places the API in test mode. This may modify behavior to facilitate testing.",
    )

    unit_test_mode: bool = Field(
        default=False,
        description="This setting only exists to facilitate unit testing. If `True`, code is executing in a unit test context. Defaults to `False`.",
    )

    unit_test_loop_debug: bool = Field(
        default=True,
        description="If `True` turns on debug mode for the unit testing event loop.",
    )

    test_setting: Optional[Any] = Field(
        default="FOO",
        description="This setting only exists to facilitate unit testing. If in test mode, this setting will return its value. Otherwise, it returns `None`.",
    )

    ###########################################################################
    # Backend API settings

    api_default_limit: int = Field(
        default=200,
        description="The default limit applied to queries that can return multiple objects, such as `POST /flow_runs/filter`.",
    )

    ###########################################################################
    # Logging settings

    logging_internal_level: LogLevel = Field(
        default="ERROR",
        description="The default logging level for Prefect's internal machinery loggers.",
    )

    ###########################################################################
    # UI settings

    ui_enabled: bool = Field(
        default=True,
        description="Whether or not to serve the Prefect UI.",
    )

    ui_url: Optional[str] = Field(
        default=None,
        description="The URL of the Prefect UI. If not set, the client will attempt to infer it.",
    )

    ui_api_url: Optional[str] = Field(
        default=None,
        description="The connection url for communication from the UI to the API. Defaults to `PREFECT_API_URL` if set. Otherwise, the default URL is generated from `PREFECT_SERVER_API_HOST` and `PREFECT_SERVER_API_PORT`.",
    )

    ui_serve_base: str = Field(
        default="/",
        description="The base URL path to serve the Prefect UI from.",
    )

    ui_static_directory: Optional[str] = Field(
        default=None,
        description="The directory to serve static files from. This should be used when running into permissions issues when attempting to serve the UI from the default directory (for example when running in a Docker container).",
    )

    ###########################################################################
    # uncategorized

    home: Annotated[Path, BeforeValidator(lambda x: Path(x).expanduser())] = Field(
        default=Path("~") / ".prefect",
        description="The path to the Prefect home directory. Defaults to ~/.prefect",
    )
    debug_mode: bool = Field(
        default=False,
        description="If True, enables debug mode which may provide additional logging and debugging features.",
    )

    silence_api_url_misconfiguration: bool = Field(
        default=False,
        description="""
        If `True`, disable the warning when a user accidentally misconfigure its `PREFECT_API_URL`
        Sometimes when a user manually set `PREFECT_API_URL` to a custom url,reverse-proxy for example,
        we would like to silence this warning so we will set it to `FALSE`.
        """,
    )

    experimental_warn: bool = Field(
        default=True,
        description="If `True`, warn on usage of experimental features.",
    )

    profiles_path: Optional[Path] = Field(
        default=None,
        description="The path to a profiles configuration file.",
    )

    tasks_refresh_cache: bool = Field(
        default=False,
        description="If `True`, enables a refresh of cached results: re-executing the task will refresh the cached results.",
    )

    task_default_retries: int = Field(
        default=0,
        ge=0,
        description="This value sets the default number of retries for all tasks.",
    )

    task_default_retry_delay_seconds: Union[int, float, list[float]] = Field(
        default=0,
        description="This value sets the default retry delay seconds for all tasks.",
    )

    sqlalchemy_pool_size: Optional[int] = Field(
        default=None,
        description="Controls connection pool size when using a PostgreSQL database with the Prefect API. If not set, the default SQLAlchemy pool size will be used.",
    )

    sqlalchemy_max_overflow: Optional[int] = Field(
        default=None,
        description="Controls maximum overflow of the connection pool when using a PostgreSQL database with the Prefect API. If not set, the default SQLAlchemy maximum overflow value will be used.",
    )

    async_fetch_state_result: bool = Field(
        default=False,
        description="""
        Determines whether `State.result()` fetches results automatically or not.
        In Prefect 2.6.0, the `State.result()` method was updated to be async
        to facilitate automatic retrieval of results from storage which means when
        writing async code you must `await` the call. For backwards compatibility,
        the result is not retrieved by default for async users. You may opt into this
        per call by passing  `fetch=True` or toggle this setting to change the behavior
        globally.
        """,
    )

    deployment_concurrency_slot_wait_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description=(
            "The number of seconds to wait before retrying when a deployment flow run"
            " cannot secure a concurrency slot from the server."
        ),
    )

    worker_heartbeat_seconds: float = Field(
        default=30,
        description="Number of seconds a worker should wait between sending a heartbeat.",
    )

    worker_query_seconds: float = Field(
        default=10,
        description="Number of seconds a worker should wait between queries for scheduled work.",
    )

    worker_prefetch_seconds: float = Field(
        default=10,
        description="The number of seconds into the future a worker should query for scheduled work.",
    )

    worker_webserver_host: str = Field(
        default="0.0.0.0",
        description="The host address the worker's webserver should bind to.",
    )

    worker_webserver_port: int = Field(
        default=8080,
        description="The port the worker's webserver should bind to.",
    )

    task_scheduling_default_storage_block: Optional[str] = Field(
        default=None,
        description="The `block-type/block-document` slug of a block to use as the default storage for autonomous tasks.",
    )

    task_scheduling_delete_failed_submissions: bool = Field(
        default=True,
        description="Whether or not to delete failed task submissions from the database.",
    )

    experimental_enable_schedule_concurrency: bool = Field(
        default=False,
        description="Whether or not to enable concurrency for scheduled tasks.",
    )

    task_runner_thread_pool_max_workers: Optional[int] = Field(
        default=None,
        gt=0,
        description="The maximum number of workers for ThreadPoolTaskRunner.",
    )

    ###########################################################################
    # allow deprecated access to PREFECT_SOME_SETTING_NAME

    def __getattribute__(self, name: str) -> Any:
        from prefect.settings.legacy import _env_var_to_accessor

        if name.startswith("PREFECT_"):
            field_name = _env_var_to_accessor(name)
            warnings.warn(
                f"Accessing `Settings().{name}` is deprecated. Use `Settings().{field_name}` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return super().__getattribute__(field_name)
        return super().__getattribute__(name)

    ###########################################################################

    @model_validator(mode="after")
    def post_hoc_settings(self) -> Self:
        """refactor on resolution of https://github.com/pydantic/pydantic/issues/9789

        we should not be modifying __pydantic_fields_set__ directly, but until we can
        define dependencies between defaults in a first-class way, we need clean up
        post-hoc default assignments to keep set/unset fields correct after instantiation.
        """
        if self.ui_url is None:
            self.ui_url = _default_ui_url(self)
            self.__pydantic_fields_set__.remove("ui_url")
        if self.ui_api_url is None:
            if self.api.url:
                self.ui_api_url = self.api.url
                self.__pydantic_fields_set__.remove("ui_api_url")
            else:
                self.ui_api_url = (
                    f"http://{self.server.api.host}:{self.server.api.port}/api"
                )
                self.__pydantic_fields_set__.remove("ui_api_url")
        if self.profiles_path is None or "PREFECT_HOME" in str(self.profiles_path):
            self.profiles_path = Path(f"{self.home}/profiles.toml")
            self.__pydantic_fields_set__.remove("profiles_path")
        if self.results.local_storage_path is None:
            self.results.local_storage_path = Path(f"{self.home}/storage")
            self.results.__pydantic_fields_set__.remove("local_storage_path")
        if self.server.memo_store_path is None:
            self.server.memo_store_path = Path(f"{self.home}/memo_store.toml")
            self.server.__pydantic_fields_set__.remove("memo_store_path")
        if self.debug_mode or self.test_mode:
            self.logging.level = "DEBUG"
            self.logging_internal_level = "DEBUG"
            self.logging.__pydantic_fields_set__.remove("level")
            self.__pydantic_fields_set__.remove("logging_internal_level")

        if self.logging.config_path is None:
            self.logging.config_path = Path(f"{self.home}/logging.yml")
            self.logging.__pydantic_fields_set__.remove("config_path")
        # Set default database connection URL if not provided
        if self.server.database.connection_url is None:
            self.server.database.connection_url = _default_database_connection_url(self)
            self.server.database.__pydantic_fields_set__.remove("connection_url")
        db_url = (
            self.server.database.connection_url.get_secret_value()
            if isinstance(self.server.database.connection_url, SecretStr)
            else self.server.database.connection_url
        )
        if (
            "PREFECT_API_DATABASE_PASSWORD" in db_url
            or "PREFECT_SERVER_DATABASE_PASSWORD" in db_url
        ):
            if self.server.database.password is None:
                raise ValueError(
                    "database password is None - please set PREFECT_SERVER_DATABASE_PASSWORD"
                )
            db_url = db_url.replace(
                "${PREFECT_API_DATABASE_PASSWORD}",
                self.server.database.password.get_secret_value()
                if self.server.database.password
                else "",
            )
            db_url = db_url.replace(
                "${PREFECT_SERVER_DATABASE_PASSWORD}",
                self.server.database.password.get_secret_value()
                if self.server.database.password
                else "",
            )
            self.server.database.connection_url = SecretStr(db_url)
            self.server.database.__pydantic_fields_set__.remove("connection_url")
        return self

    @model_validator(mode="after")
    def emit_warnings(self) -> Self:
        """More post-hoc validation of settings, including warnings for misconfigurations."""
        values = self.model_dump()
        if not self.silence_api_url_misconfiguration:
            values = _warn_on_misconfigured_api_url(values)
        return self

    ##########################################################################
    # Settings methods

    def copy_with_update(
        self: Self,
        updates: Optional[Mapping["Setting", Any]] = None,
        set_defaults: Optional[Mapping["Setting", Any]] = None,
        restore_defaults: Optional[Iterable["Setting"]] = None,
    ) -> Self:
        """
        Create a new Settings object with validation.

        Arguments:
            updates: A mapping of settings to new values. Existing values for the
                given settings will be overridden.
            set_defaults: A mapping of settings to new default values. Existing values for
                the given settings will only be overridden if they were not set.
            restore_defaults: An iterable of settings to restore to their default values.

        Returns:
            A new Settings object.
        """
        restore_defaults_obj = {}
        for r in restore_defaults or []:
            set_in_dict(restore_defaults_obj, r.accessor, True)
        updates = updates or {}
        set_defaults = set_defaults or {}

        set_defaults_obj = {}
        for setting, value in set_defaults.items():
            set_in_dict(set_defaults_obj, setting.accessor, value)

        updates_obj = {}
        for setting, value in updates.items():
            set_in_dict(updates_obj, setting.accessor, value)

        new_settings = self.__class__.model_validate(
            deep_merge_dicts(
                set_defaults_obj,
                self.model_dump(exclude_unset=True, exclude=restore_defaults_obj),
                updates_obj,
            )
        )
        return new_settings

    def hash_key(self) -> str:
        """
        Return a hash key for the settings object.  This is needed since some
        settings may be unhashable, like lists.
        """
        env_variables = self.to_environment_variables()
        return str(hash(tuple((key, value) for key, value in env_variables.items())))


def _default_ui_url(settings: "Settings") -> Optional[str]:
    value = settings.ui_url
    if value is not None:
        return value

    # Otherwise, infer a value from the API URL
    ui_url = api_url = settings.api.url

    if not api_url:
        return None
    assert ui_url is not None

    cloud_url = settings.cloud.api_url
    cloud_ui_url = settings.cloud.ui_url
    if api_url.startswith(cloud_url) and cloud_ui_url:
        ui_url = ui_url.replace(cloud_url, cloud_ui_url)

    if ui_url.endswith("/api"):
        # Handles open-source APIs
        ui_url = ui_url[:-4]

    # Handles Cloud APIs with content after `/api`
    ui_url = ui_url.replace("/api/", "/")

    # Update routing
    ui_url = ui_url.replace("/accounts/", "/account/")
    ui_url = ui_url.replace("/workspaces/", "/workspace/")

    return ui_url


def _warn_on_misconfigured_api_url(values):
    """
    Validator for settings warning if the API URL is misconfigured.
    """
    api_url = values.get("api", {}).get("url")
    if api_url is not None:
        misconfigured_mappings = {
            "app.prefect.cloud": (
                "`PREFECT_API_URL` points to `app.prefect.cloud`. Did you"
                " mean `api.prefect.cloud`?"
            ),
            "account/": (
                "`PREFECT_API_URL` uses `/account/` but should use `/accounts/`."
            ),
            "workspace/": (
                "`PREFECT_API_URL` uses `/workspace/` but should use `/workspaces/`."
            ),
        }
        warnings_list = []

        for misconfig, warning in misconfigured_mappings.items():
            if misconfig in api_url:
                warnings_list.append(warning)

        parsed_url = urlparse(api_url)
        if parsed_url.path and not parsed_url.path.startswith("/api"):
            warnings_list.append(
                "`PREFECT_API_URL` should have `/api` after the base URL."
            )

        if warnings_list:
            example = 'e.g. PREFECT_API_URL="https://api.prefect.cloud/api/accounts/[ACCOUNT-ID]/workspaces/[WORKSPACE-ID]"'
            warnings_list.append(example)

            warnings.warn("\n".join(warnings_list), stacklevel=2)

    return values


def _default_database_connection_url(settings: "Settings") -> SecretStr:
    value = None
    if settings.server.database.driver == "postgresql+asyncpg":
        required = [
            "host",
            "user",
            "name",
            "password",
        ]
        missing = [
            attr for attr in required if getattr(settings.server.database, attr) is None
        ]
        if missing:
            raise ValueError(
                f"Missing required database connection settings: {', '.join(missing)}"
            )

        from sqlalchemy import URL

        return URL(
            drivername=settings.server.database.driver,
            host=settings.server.database.host,
            port=settings.server.database.port or 5432,
            username=settings.server.database.user,
            password=(
                settings.server.database.password.get_secret_value()
                if settings.server.database.password
                else None
            ),
            database=settings.server.database.name,
            query=[],  # type: ignore
        ).render_as_string(hide_password=False)

    elif settings.server.database.driver == "sqlite+aiosqlite":
        if settings.server.database.name:
            value = (
                f"{settings.server.database.driver}:///{settings.server.database.name}"
            )
        else:
            value = f"sqlite+aiosqlite:///{settings.home}/prefect.db"

    elif settings.server.database.driver:
        raise ValueError(
            f"Unsupported database driver: {settings.server.database.driver}"
        )

    value = value if value else f"sqlite+aiosqlite:///{settings.home}/prefect.db"
    return SecretStr(value)