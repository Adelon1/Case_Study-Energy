"""Project-wide constants and ENTSO-E dataset request definitions.

The project needs a few stable values in many places: the ENTSO-E API URL,
Germany's bidding-zone code, timestamp formats, and the mapping from friendly
dataset names to ENTSO-E query parameters. Keeping those values here makes the
pipeline easier to audit and change.
"""

from __future__ import annotations

from dataclasses import dataclass


ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"

# Germany/Luxembourg bidding zone used for current German day-ahead prices.
# Historical periods before the 2018 bidding-zone split may need DE-AT-LU.
GERMANY_LUXEMBOURG_BIDDING_ZONE_EIC = "10Y1001A1001A82H"

GERMANY_MARKET_TIMEZONE = "Europe/Berlin"
ENTSOE_DATETIME_FORMAT = "%Y%m%d%H%M"


@dataclass(frozen=True)
class EntsoeDatasetRequest:
    """Named ENTSO-E API request template.

    ``params`` contains the ENTSO-E query parameters except token and date
    range. ``value_tag`` and ``output_column`` tell the XML parser what to read
    and how the final CSV column should be named.
    """

    name: str
    description: str
    params: dict[str, str]
    value_tag: str
    output_column: str
    required_time_series_tags: dict[str, str]


ENTSOE_DATASETS: dict[str, EntsoeDatasetRequest] = {
    "day_ahead_prices": EntsoeDatasetRequest(
        name="day_ahead_prices",
        description="Sequence 1 SDAC day-ahead electricity prices for DE-LU.",
        params={
            "documentType": "A44",
            "in_Domain": GERMANY_LUXEMBOURG_BIDDING_ZONE_EIC,
            "out_Domain": GERMANY_LUXEMBOURG_BIDDING_ZONE_EIC,
            "contract_MarketAgreement.type": "A01",
        },
        value_tag="price.amount",
        output_column="day_ahead_price_eur_per_mwh",
        required_time_series_tags={
            "classificationSequence_AttributeInstanceComponent.position": "1",
        },
    ),
    "load_forecast": EntsoeDatasetRequest(
        name="load_forecast",
        description="Day-ahead total load forecast for DE-LU.",
        params={
            "documentType": "A65",
            "processType": "A01",
            "outBiddingZone_Domain": GERMANY_LUXEMBOURG_BIDDING_ZONE_EIC,
        },
        value_tag="quantity",
        output_column="load_forecast_mw",
        required_time_series_tags={},
    ),
    "load_actual": EntsoeDatasetRequest(
        name="load_actual",
        description="Actual total load for DE-LU. Use only lagged values in live forecasts.",
        params={
            "documentType": "A65",
            "processType": "A16",
            "outBiddingZone_Domain": GERMANY_LUXEMBOURG_BIDDING_ZONE_EIC,
        },
        value_tag="quantity",
        output_column="load_actual_mw",
        required_time_series_tags={},
    ),
    "solar_forecast": EntsoeDatasetRequest(
        name="solar_forecast",
        description="Day-ahead solar generation forecast for DE-LU.",
        params={
            "documentType": "A69",
            "processType": "A01",
            "in_Domain": GERMANY_LUXEMBOURG_BIDDING_ZONE_EIC,
            "psrType": "B16",
        },
        value_tag="quantity",
        output_column="solar_forecast_mw",
        required_time_series_tags={},
    ),
    "wind_onshore_forecast": EntsoeDatasetRequest(
        name="wind_onshore_forecast",
        description="Day-ahead onshore wind generation forecast for DE-LU.",
        params={
            "documentType": "A69",
            "processType": "A01",
            "in_Domain": GERMANY_LUXEMBOURG_BIDDING_ZONE_EIC,
            "psrType": "B19",
        },
        value_tag="quantity",
        output_column="wind_onshore_forecast_mw",
        required_time_series_tags={},
    ),
    "wind_offshore_forecast": EntsoeDatasetRequest(
        name="wind_offshore_forecast",
        description="Day-ahead offshore wind generation forecast for DE-LU.",
        params={
            "documentType": "A69",
            "processType": "A01",
            "in_Domain": GERMANY_LUXEMBOURG_BIDDING_ZONE_EIC,
            "psrType": "B18",
        },
        value_tag="quantity",
        output_column="wind_offshore_forecast_mw",
        required_time_series_tags={},
    ),
}


def get_entsoe_dataset_request(name: str) -> EntsoeDatasetRequest:
    """Return the ENTSO-E request template for a friendly dataset name."""

    try:
        return ENTSOE_DATASETS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(ENTSOE_DATASETS))
        raise ValueError(f"Unknown ENTSO-E dataset '{name}'. Valid options: {valid}") from exc
