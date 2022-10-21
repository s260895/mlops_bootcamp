from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import SubProcessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main
    name=IntervalSchedule(interval=timedelta(minutes=5)),
    flow_runner = SubProcessFlowRunner(),
    tags=['ml']
)

