from pandas import DataFrame
from .utils import (
    PosCat,
    append_avg_position,
    append_last_n_races,
    append_avg_position_move,
    append_position_propensity,
    get_position_category,
)
from ..preprocessing import merge_datasets


def drop_features(df: DataFrame) -> DataFrame:
    return df.drop(
        [
            "raceId",
            "driverId",
            "constructorId",
            "position",
            "points",
            "wins",
            "position_numeric",
            "year",
            "circuitId",
        ],
        axis=1,
    )


def get_dataset(pos_cat: PosCat = "tight") -> DataFrame:
    df = merge_datasets()
    df["positionText"] = df["positionText"].apply(
        lambda x: get_position_category(x, pos_cat)
    )
    df = append_last_n_races(df, 10, "positionText")
    df = append_avg_position_move(df, in_season=False, progressive=True, window=10)
    # df = append_avg_position(df, in_season=True, progressive=True, window=5)
    # df = append_position_propensity(df, pos_cat, in_season=True)
    return df
