from typing import Optional
from pandas import DataFrame
from .utils import (
    PosCat,
    append_avg_quali_position_move,
    append_circuit_encodings,
    append_confidence,
    append_drivers_age,
    append_elo_change,
    append_elo_percentile,
    append_elo_rank_in_race,
    append_elo,
    append_field_pos_delta,
    append_dnf_count,
    append_last_n,
    append_avg_position_move,
    append_last_n_podiums,
    append_last_season_wins,
    append_median_race_position,
    append_nationality_encodings,
    append_position_propensity,
    append_sma,
    append_std,
    append_constructor_encodings,
    get_position_category,
)
from ..preprocessing import merge_datasets


def drop_features(df: DataFrame) -> DataFrame:
    return df.drop(
        [
            "circuitId",
            "constructorId",
            "driverId",
            "statusId",
            "raceId",
            "position",
            "points",
            "wins",
            "position_numeric",
            "positionText",
            "position_driver_standings",
            "points_constructor_standings",
            "position_constructor_standings",
            "positionOrder",
            "constructorRef",
            "driverRef",
            "circuitRef",
            "year",
            "round",
            "dob",
            "nationality",
            "prev_points_constructor_standings",
            ####
            "prev_position_driver_standings",
        ],
        axis=1,
    )


def get_dataset(pos_cat: Optional[PosCat] = None) -> DataFrame:
    """
    Returns DataFrame comprised of all necessary features for training
    or testing.

    Args:
        pos_cat (Optional[PosCat], optional): If passed, the series within the
        target column is the positionText categorised into *pos_cat* category
        Else the target is the positionOrder. This is done as it's assumed
        you're performing regression otherwise. Defaults to None.

    Returns:
        DataFrame: DataFrame comprised of all features.
    """
    df: DataFrame = merge_datasets()

    if pos_cat is None:
        df["target"] = df["positionOrder"]
    else:
        df["target"] = df["positionText"].apply(
            lambda x: get_position_category(x, pos_cat)
        )

    w = 1

    df = append_elo(df, k=200, p=0.01)
    df = append_elo_change(df)
    # df = append_elo_rank_in_race(df)
    df = append_last_n_podiums(df, window=0)
    # df = append_constructor_encodings(df)
    return df


if __name__ == "__main__":
    df = get_dataset("winner")
    # df = df.groupby("driverId").filter(lambda x: (x["elo"] < 0).any())
    # df = df[df["driverId"] == df["driverId"].iloc[0]]
    df = drop_features(df)
    df.to_csv("file.csv", index=False)
