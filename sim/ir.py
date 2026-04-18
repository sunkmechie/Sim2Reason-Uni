from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PoseIR:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class GeomIR:
    geom_id: str
    name: str
    geom_type: str | None
    pose: PoseIR
    size: tuple[float, ...] | None = None
    material: str | None = None
    rgba: tuple[float, float, float, float] | None = None
    mass: float | None = None
    backend_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SiteIR:
    site_id: str
    name: str
    pose: PoseIR
    body_name: str | None = None
    role: str | None = None
    backend_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JointIR:
    joint_id: str
    name: str
    joint_type: str
    axis: tuple[float, float, float]
    pose: PoseIR
    limit: tuple[float, float] | None = None
    damping: float | None = None
    stiffness: float | None = None
    backend_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TendonAnchorIR:
    anchor_id: str
    anchor_type: str
    body_name: str | None = None
    backend_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TendonIR:
    tendon_id: str
    name: str
    segments: tuple[tuple[TendonAnchorIR, ...], ...]
    stiffness: float | None = None
    spring_length: float | None = None
    damping: float | None = None
    is_spring: bool = False
    backend_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BodyIR:
    body_id: str
    name: str
    body_type: str
    pose: PoseIR
    geoms: tuple[GeomIR, ...] = ()
    sites: tuple[SiteIR, ...] = ()
    joints: tuple[JointIR, ...] = ()
    children: tuple["BodyIR", ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    backend_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntityIR:
    entity_id: str
    name: str
    entity_type: str
    pose: PoseIR
    bodies: tuple[BodyIR, ...]
    parameters: dict[str, Any] = field(default_factory=dict)
    labels: tuple[str, ...] = ()
    backend_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConnectionEndpointIR:
    entity_name: str
    direction: str
    connecting_point: str
    connecting_point_seq_id: int | None = None


@dataclass(slots=True)
class ConnectionIR:
    connection_id: str
    tendon: tuple[ConnectionEndpointIR, ...]
    backend_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SensorIR:
    sensor_id: str
    name: str
    sensor_type: str
    target: str | None = None
    backend_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActuatorIR:
    actuator_id: str
    name: str
    actuator_type: str
    target: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    backend_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneIR:
    scene_id: str
    name: str
    version: str = "v1"
    gravity: tuple[float, float, float] | float = -9.81
    units: str = "si"
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: tuple[EntityIR, ...] = ()
    connections: tuple[ConnectionIR, ...] = ()
    tendons: tuple[TendonIR, ...] = ()
    sensors: tuple[SensorIR, ...] = ()
    actuators: tuple[ActuatorIR, ...] = ()
    labels: tuple[str, ...] = ()
    assets: tuple[Any, ...] = ()
    backend_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
