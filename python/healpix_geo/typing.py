from typing import NotRequired, Protocol, TypedDict


class SphereDict(TypedDict):
    name: NotRequired[str]
    radius: float


class SphereType(Protocol):
    radius: float
    name: str | None


class EllipsoidDict(TypedDict):
    name: NotRequired[str]
    semimajor_axis: float
    inverse_flattening: float


class EllipsoidType(Protocol):
    semimajor_axis: float
    inverse_flattening: float
    name: str | None


_SphereLike = SphereDict | SphereType
_EllipsoidLike = EllipsoidDict | EllipsoidType

EllipsoidLike = str | _SphereLike | _EllipsoidLike
