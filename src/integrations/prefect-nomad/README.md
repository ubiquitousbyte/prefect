# prefect-nomad

<p align="center">
    <a href="https://pypi.python.org/pypi/prefect-nomad/" alt="PyPI version">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/prefect-nomad?color=26272B&labelColor=090422"></a>
    <a href="https://pepy.tech/badge/prefect-nomad/" alt="Downloads">
        <img src="https://img.shields.io/pypi/dm/prefect-nomad?color=26272B&labelColor=090422" /></a>
</p>

Visit the full docs at [https://docs.prefect.io/integrations/prefect-nomad](https://docs.prefect.io/integrations/prefect-nomad) to see additional examples and the API reference.

The `prefect-nomad` integration enables orchestration and execution of Prefect flow runs as batch jobs on HashiCorp Nomad clusters using the Docker task driver.

## Getting started

### Prerequisites

- [Nomad cluster](https://www.nomadproject.io/) running and accessible
- Docker driver configured on Nomad client nodes

### Installation

Install or update to the latest version of the Prefect Nomad integration:

```bash
pip install -U prefect-nomad
```

### Register a Nomad work pool

Create a Nomad work pool to route flow runs to your Nomad cluster:

```bash
prefect work-pool create --type nomad my-nomad-pool
```

### Start a Nomad worker

Start a worker to execute flow runs from the work pool:

```bash
prefect worker start --pool my-nomad-pool
```

## Resources

For assistance with Nomad, consult the [Nomad documentation](https://www.nomadproject.io/docs).

Refer to the prefect-nomad SDK documentation linked above to explore all capabilities of this integration.
