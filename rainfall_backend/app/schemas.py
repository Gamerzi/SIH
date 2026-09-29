"""Pydantic request models."""
from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    nwp_rain: float = Field(..., ge=0, description="Raw NWP rainfall forecast (mm/day)")
    temp: Optional[float] = None
    humidity: Optional[float] = None
    wind: Optional[float] = None
    pressure: Optional[float] = None
    cape: Optional[float] = None
    terrain_height: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    month: Optional[int] = Field(None, ge=1, le=12)
    actual_rain: Optional[float] = Field(None, ge=0, description="Observed rain, only for plots")

    def features(self) -> dict:
        d = self.model_dump(exclude={"actual_rain"})
        return {k: v for k, v in d.items() if v is not None}
