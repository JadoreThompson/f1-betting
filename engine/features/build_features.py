from pandas import DataFrame
from .utils import (
    PosCat,
    append_field_pos_delta,
    append_dnf_count,
    append_last_n_races,
    append_avg_position_move,
    append_last_season_wins,
    append_median_race_position,
    append_position_propensity,
    append_sma,
    get_position_category,
)
from ..preprocessing import merge_datasets


def drop_features(df: DataFrame) -> DataFrame:
    return df.drop(
        [
            "circuitId",
            "constructorId",
            "driverId",
            "raceId",
            "position",
            "points",
            "wins",
            "position_numeric",
            "positionText",
            "position_standings",
            "statusId",
            "year",
        ],
        axis=1,
    )


def get_dataset(pos_cat: PosCat) -> DataFrame:
    df: DataFrame = merge_datasets()
    df["target"] = df["positionText"].apply(lambda x: get_position_category(x, pos_cat))
    df = append_last_n_races(df, "target", in_season=True, window=3)
    df = append_position_propensity(df, pos_cat, in_season=True)
    df = append_median_race_position(df, window=1)
    return df


# get_dataset("loose").to_csv("file.csv", index=False)
